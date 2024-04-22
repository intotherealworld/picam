import argparse
import glob
import importlib
import os

import uvicorn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summer local server runner')
    parser.add_argument('--module', help='root module name. optional.', type=str, default=None, required=False)
    parser.add_argument('--port', help='port, default 5000, optional.', type=int, default=5000, required=False)
    parser.add_argument('--no-reload',
                        help='disable automatic reload when a code changes. optional',
                        required=False,
                        action='store_true')
    args = parser.parse_args()
    module_name = args.module
    if not module_name:
        yml_path = 'config/properties.yml'
        module_name = None
        for d in filter(lambda f: os.path.isdir(f), glob.glob('*')):
            if os.path.isfile(f'{d}/{yml_path}'):
                module_name = d
                break

    module = importlib.import_module(module_name)
    getattr(module, 'create_app')

    reload = True if args.no_reload is None else False

    uvicorn.run(f'{module_name}:create_app', host='0.0.0.0', port=args.port, reload=reload)
