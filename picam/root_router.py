import os.path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from summer_toolkit.framework.simple_jinja2_templates import SimpleJinja2Templates

root_router = APIRouter(tags=['root'])
templates = SimpleJinja2Templates()


class DefaultResponse(BaseModel):
    status: str = Field('ok')

    class Config:
        schema_extra = {
            'example': {
                'status': 'ok'
            }
        }


@root_router.get('/', include_in_schema=False)
def respond_root(request: Request):
    return templates.TemplateResponse('root.html', {'request': request})


@root_router.get('/favicon.ico', include_in_schema=False)
def respond_favicon():
    return FileResponse(f'{os.path.dirname(os.path.realpath(__file__))}/static/favicon.ico')


@root_router.get('/hc', include_in_schema=False)
def respond_health_check():
    return DefaultResponse(status='ok')
