
## Local Development

Run python setup to install the theme:

```
python setup.py install
```

To install the dependencies for this theme, run:

```
pip install -r dependencies/requirements.txt
```


In the root directory install the dependencies of `package.json`:

```
# node version 8.4.0
npm install
```

Now we can run the generated docs in localhost:1919 using :

```
grunt

```

### Grunt options

- 'grunt --project=docs'

This will first look for the path of the numpy/doc in .env file. To make it
work we first need to speficy the path of the numpy source folder.

Example: If you have placed the numpy source code in the same directory of the
scipy-sphinx-theme-v2 then `.env` file will have:

```
{

"DOCS_DIR":"../numpy/doc/source"

}

```

- 'grunt build'


Using this command sphinx build command will run for the project dir on the build directory.

Default project directory is `demo-docs`. You can specify the project directory using `grunt --project=docs build`. Now the new project directory
will be whatever is set in your .env file.

Build directory will be `project_dir/build`.

- 'grunt serve'

If you have already build the HTML files and just want to get it live on `localhost:2121` (means skip the build part), then run `grunt serve`.

This command will serve the build directory on `localhost:2121`.


**Note:**

- Sample docs is present on demo-docs folder.
- grunt will automatically refresh the page when we do changes in the docs file.

[TODO]

- Run the docs folder
- Run the devdocs folder
- Run the generatd numpy docs

## Surge deploy

- Every PR will be deployed on surge automatically.
- URL will be pr-<pr_number>-scipy-sphinx-theme-v2.surge.sh
- For example: PR #3 is deployed on https://pr-3-scipy-sphinx-theme-v2.surge.sh
