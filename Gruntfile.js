module.exports = function(grunt) {
  // load all grunt tasks
  require('matchdep').filterDev('grunt-*').forEach(grunt.loadNpmTasks);

  var envJSON = grunt.file.readJSON(".env.json");
  var PROJECT_DIR = "demo-docs/";

  switch (grunt.option('project')) {
    case "docs":
      PROJECT_DIR = envJSON.DOCS_DIR;
      break;
    case "devdocs":
      PROJECT_DIR = envJSON.DEVDOCS_DIR;
      break;
   }

  grunt.initConfig({
    // Read package.json
    pkg: grunt.file.readJSON("package.json"),

    open : {
      dev: {
        path: 'http://localhost:2121'
      }
    },

    connect: {
      server: {
        options: {
          port: 2121,
          base: 'demo-docs/build',
          livereload: true
        }
      }
    },
    copy: {

      vendor: {
        files: [
          {
              expand: true,
              cwd: 'node_modules/bootstrap/scss/',
              src: "**/*",
              dest: 'dependencies/scss/vendor/bootstrap',
              filter: 'isFile'
          },

          {
            expand: true,
            flatten: true,
            src: [
              'node_modules/popper.js/dist/umd/popper.min.js',
              'node_modules/bootstrap/dist/js/bootstrap.min.js',
              'node_modules/anchor-js/anchor.min.js'
            ],
            dest: 'dependencies/static/js/vendor',
            filter: 'isFile'
          }
        ]
      }
    },

    exec: {
      build_sphinx: {
        cmd: 'sphinx-build ' + PROJECT_DIR + ' demo-docs/build'
      }
    },
    clean: {
      build: ["demo-docs/build"],
    },

    watch: {

      /* Changes in theme dir rebuild sphinx */
      sphinx: {
        files: ['custom_sphinx_theme/**/*', 'README.rst', 'demo-docs/**/*.rst', 'demo-docs/**/*.py'],
        tasks: ['clean:build','exec:build_sphinx']
      },
      /* JavaScript */
      browserify: {
        files: ['js/*.js'],
        tasks: ['browserify:dev']
      },
      /* live-reload the docs if sphinx re-builds */
      livereload: {
        files: ['demo-docs/build/**/*'],
        options: { livereload: true }
      }
    },
    surge: {
      'scipy-sphinx-theme-v2': {
        options: {
          // The path or directory to your compiled project
          project: 'demo-docs/build/',
          // The domain or subdomain to deploy to
          domain: grunt.option('domain')
        }
      }
    }

  });

  grunt.loadNpmTasks('grunt-exec');
  grunt.loadNpmTasks('grunt-contrib-connect');
  grunt.loadNpmTasks('grunt-contrib-watch');
  grunt.loadNpmTasks('grunt-contrib-sass');
  grunt.loadNpmTasks('grunt-contrib-clean');
  grunt.loadNpmTasks('grunt-contrib-copy');
  grunt.loadNpmTasks('grunt-open');
  grunt.loadNpmTasks('grunt-browserify');

  grunt.registerTask('default', ['clean','copy:vendor','exec:build_sphinx','connect','open','watch']);
  grunt.registerTask('build', ['clean','copy:fonts', 'copy:images', 'copy:vendor', 'sass:build', 'postcss:dist', 'browserify:build', 'uglify']);
  grunt.registerTask('deploy', ['clean','exec:build_sphinx','surge']);
}
