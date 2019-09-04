
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
