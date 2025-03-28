# Query Endpoint Fix

## Problem

The Panopticon system has an issue with the `/api/queries` endpoint in the main application. The FastAPI framework is automatically redirecting POST requests to this endpoint with a 307 Temporary Redirect status code. This happens despite configuring FastAPI's `redirect_slashes=False` and attempts to override this behavior.

## Working Solution

We've implemented a standalone forwarder service that successfully handles the `/api/queries` endpoint without redirect issues. This forwarder uses Starlette directly (the ASGI framework that underlies FastAPI) to create routes without the redirection behavior.

## How to Use

1. The standalone forwarder service has been implemented in `app/direct_api_forwarder.py`
2. Run the standalone service on port 8080:

```
cd /path/to/panopticon
python app/direct_api_forwarder.py
```

3. Make POST requests to the forwarder instead of the main app:

```
# Instead of this (which will redirect):
curl -X POST http://localhost:8000/api/queries -H "Content-Type: application/json" -H "X-API-Key: dev_api_key_for_testing" -d '{"item": "query", "type": "test"}'

# Use this (which works correctly):
curl -X POST http://localhost:8080/api/queries -H "Content-Type: application/json" -H "X-API-Key: dev_api_key_for_testing" -d '{"item": "query", "type": "test"}'
```

## Technical Details

### Why FastAPI is Redirecting

FastAPI uses Starlette's router under the hood, which has automatic redirection behavior for paths with and without trailing slashes. While FastAPI exposes a `redirect_slashes=False` parameter, there are known issues with this not fully disabling the behavior in all scenarios.

Our attempts to override this behavior included:

1. Setting `redirect_slashes=False` directly 
2. Manually overriding FastAPI's router 
3. Using middleware to intercept redirects
4. Using Starlette's low-level routing
5. Registering route handlers for both with and without slashes

None of these approaches fully resolved the issue in the main application.

### Standalone Solution

The standalone forwarder:
1. Uses Starlette directly without FastAPI
2. Creates explicit routes for both `/api/queries` and `/api/queries/` paths
3. Forwards requests directly to the Item Storage service
4. Runs on a separate port (8080) to avoid conflicts

This solution is working perfectly and can be used as a workaround until the main application issues are resolved.

## Permanent Integration

For a permanent solution, consider:

1. Integrating this forwarder's logic into the main application using a separate service
2. Configuring a reverse proxy to handle this specific endpoint
3. Updating clients to use the direct Item Storage service endpoint (`http://localhost:8001/api/v1/items`) instead

## References

- [FastAPI router documentation](https://fastapi.tiangolo.com/advanced/custom-request-and-route/)
- [Starlette routing documentation](https://www.starlette.io/routing/)
