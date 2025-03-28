"""
Database utilities for all microservices in the Panopticon system.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from sqlalchemy import create_engine, Engine, Table, text, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import logging
from contextlib import contextmanager, asynccontextmanager

# Base class for all models
Base = declarative_base()

# Type variable for generic functions
T = TypeVar('T')

class Database:
    """Database connection and session management."""
    
    def __init__(self, connection_string: str, pool_size: int = 5, max_overflow: int = 10):
        """
        Initialize database connection.
        
        Args:
            connection_string: SQLAlchemy connection string
            pool_size: Connection pool size
            max_overflow: Maximum number of connections to overflow
        """
        # Synchronous engine
        self.engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True  # Check connection before using from pool
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Asynchronous engine (for asyncpg)
        async_connection_string = connection_string.replace('postgresql://', 'postgresql+asyncpg://')
        self.async_engine = create_async_engine(
            async_connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True
        )
        self.async_session_factory = sessionmaker(
            class_=AsyncSession, 
            expire_on_commit=False, 
            bind=self.async_engine
        )
        
        self.logger = logging.getLogger("database")
    
    def create_tables(self) -> None:
        """Create all tables defined in Base."""
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.exception("Database error: %s", str(e))
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncSession:
        """Get an async database session with automatic cleanup."""
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self.logger.exception("Async database error: %s", str(e))
            raise
        finally:
            await session.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries with query results
        """
        with self.engine.connect() as connection:
            result = connection.execute(text(query), params or {})
            return [dict(row) for row in result]
    
    def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error("Database connection check failed: %s", str(e))
            return False
    
    async def check_async_connection(self) -> bool:
        """Check if async database connection is working."""
        try:
            async with self.async_engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error("Async database connection check failed: %s", str(e))
            return False

# Repository pattern base class
class Repository(Generic[T]):
    """Base class for repositories implementing CRUD operations."""
    
    def __init__(self, db: Database, model_class: Type[T]):
        """
        Initialize repository.
        
        Args:
            db: Database instance
            model_class: SQLAlchemy model class
        """
        self.db = db
        self.model_class = model_class
    
    def create(self, **kwargs) -> T:
        """Create a new record."""
        with self.db.get_session() as session:
            obj = self.model_class(**kwargs)
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj
    
    async def create_async(self, **kwargs) -> T:
        """Create a new record asynchronously."""
        async with self.db.get_async_session() as session:
            obj = self.model_class(**kwargs)
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
            return obj
    
    def get_by_id(self, id: Any) -> Optional[T]:
        """Get record by ID."""
        with self.db.get_session() as session:
            return session.query(self.model_class).filter(self.model_class.id == id).first()
    
    async def get_by_id_async(self, id: Any) -> Optional[T]:
        """Get record by ID asynchronously."""
        async with self.db.get_async_session() as session:
            stmt = select(self.model_class).where(self.model_class.id == id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    def list(self, skip: int = 0, limit: int = 100, **filters) -> List[T]:
        """List records with optional filtering."""
        with self.db.get_session() as session:
            query = session.query(self.model_class)
            
            # Apply filters
            for attr, value in filters.items():
                if hasattr(self.model_class, attr):
                    query = query.filter(getattr(self.model_class, attr) == value)
            
            return query.offset(skip).limit(limit).all()
    
    async def list_async(self, skip: int = 0, limit: int = 100, **filters) -> List[T]:
        """List records with optional filtering asynchronously."""
        async with self.db.get_async_session() as session:
            stmt = select(self.model_class)
            
            # Apply filters
            for attr, value in filters.items():
                if hasattr(self.model_class, attr):
                    stmt = stmt.where(getattr(self.model_class, attr) == value)
            
            # Add offset and limit
            stmt = stmt.offset(skip).limit(limit)
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    def update(self, id: Any, **kwargs) -> Optional[T]:
        """Update record by ID."""
        with self.db.get_session() as session:
            obj = session.query(self.model_class).filter(self.model_class.id == id).first()
            if obj:
                for key, value in kwargs.items():
                    setattr(obj, key, value)
                session.commit()
                session.refresh(obj)
            return obj
    
    async def update_async(self, id: Any, **kwargs) -> Optional[T]:
        """Update record by ID asynchronously."""
        async with self.db.get_async_session() as session:
            stmt = select(self.model_class).where(self.model_class.id == id)
            result = await session.execute(stmt)
            obj = result.scalar_one_or_none()
            
            if obj:
                for key, value in kwargs.items():
                    setattr(obj, key, value)
                await session.commit()
                await session.refresh(obj)
            return obj
    
    def delete(self, id: Any) -> bool:
        """Delete record by ID."""
        with self.db.get_session() as session:
            obj = session.query(self.model_class).filter(self.model_class.id == id).first()
            if obj:
                session.delete(obj)
                session.commit()
                return True
            return False
    
    async def delete_async(self, id: Any) -> bool:
        """Delete record by ID asynchronously."""
        async with self.db.get_async_session() as session:
            stmt = select(self.model_class).where(self.model_class.id == id)
            result = await session.execute(stmt)
            obj = result.scalar_one_or_none()
            
            if obj:
                await session.delete(obj)
                await session.commit()
                return True
            return False
    
    def count(self, **filters) -> int:
        """Count records with optional filtering."""
        with self.db.get_session() as session:
            query = session.query(self.model_class)
            
            # Apply filters
            for attr, value in filters.items():
                if hasattr(self.model_class, attr):
                    query = query.filter(getattr(self.model_class, attr) == value)
            
            return query.count()
    
    async def count_async(self, **filters) -> int:
        """Count records with optional filtering asynchronously."""
        async with self.db.get_async_session() as session:
            stmt = select(self.model_class)
            
            # Apply filters
            for attr, value in filters.items():
                if hasattr(self.model_class, attr):
                    stmt = stmt.where(getattr(self.model_class, attr) == value)
            
            result = await session.execute(stmt)
            return len(result.scalars().all())
