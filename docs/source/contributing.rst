==================================================
Contributing
==================================================

Feedback
------------

Did you find a bug in AWEbox, or do you think the tool is missing some possibly very useful feature?

Please have a look at our issue tracker (https://github.com/awebox/awebox/issues) to check whether the issue already exists.
If not, create an issue with a detailed description of the problem. If it already exists, feel free to weigh in on the discussion or to simply state how important this issue is for your personal work.

Note that your issue might also be caused by problems related to CasADi, the underlying symbolic language that AWEbox uses. 
If you think this might be the case, have a look at the CasADi `forum <https://groups.google.com/forum/?fromgroups=#!forum/casadi-users>`_ as well.

If you are interested in a closer collaboration with the core development team: we are open for academic cooperation. In this case, please contact one the maintainers personally.

Pull requests
---------------

AWEbox is developed following the `git flow <https://nvie.com/posts/a-successful-git-branching-model/>`_ branching model. Read it. Use it.

Did you implement a new feature or fix a bug for AWEbox? 
Below is a checklist that will help you get your Pull Request (PR) accepted:

- Create a Pull Request *early* with the prefix "[WIP]", so that others can track your progress and give you some early feedback.
- Cross-reference your PR to an Issue in the Issues Tracker.
- Implement new functionality in several *small* features. Features should be finished within a couple of days at most. 
- Make sure that your change does not break anything by testing your code:

    .. code-block:: bash

        $ cd test/
        $ python3 -m pytest
    
  In case you're adding new functionality: write a new test!

- Inspect your code cleanliness by running pylint in the root folder:

    .. code-block:: bash

        $ pylint -d w -d c -d r /awebox 

- Write docstrings for new modules/classes/methods or update them if necessary (see the `Documentation`_ section below).
- Add a ChangeLog entry, short for bugfixes, more elaborate for new features. (will be added upon release)
- Add yourself to the Contributors file, if not already the case. (will be added upon release)
- Make relevant changes to the documentation (in the ``docs/source/`` folder).
- Remove the "[WIP]"-prefix and assign someone to review your PR.

Documentation
--------------------------

How to contribute:

1. Install Sphinx:

    .. code-block:: bash
       
        $ apt-get install python3-sphinx
    

2. Add or edit a documentation page by navigating to ``docs/source/`` and by editing one of the .rst-files (e.g. ``introduction.rst``). 
   In case you add a page, don't forget to add it to the table of contents in ``index.rst``. 
   Hold in mind that these pages are written in reStructuredText markup language; 
   some IDEs have plugins that allow you to have a preview of the resulting layout.
 
3. Write docstrings when writing new modules/classes/methods. Use the reStructuredText-format for docstrings. Example:

    .. code-block:: python

        class Dae(object):
            """
            Dae object that serves as an interface to CasADi's
            `rootfinder <http://casadi.sourceforge.net/api/html/d3/d65/group__rootfinder.html>`_ and
            `integrator <http://casadi.sourceforge.net/api/html/dd/d1b/group__integrator.html>`_ 
            solvers for awebox Model objects .
            """

            def __init__(self, variables, parameters, dynamics, integral_outputs_fun):
                """ Constructor.
            
                :type variables: casadi.tools.structure.ssymStruct
                :param variables: model variables
            
                :type parameters: casadi.tools.structure.ssymStruct
                :param parameters: model parameters
            
                :type dynamics: casadi.Function
                :param dynamics: fully implicit dae dynamics
            
                :type integral_outputs_fun: casadi.Function
                :param integral_outputs_fun: quadrature state dynamics
            
                :raises ValueError: if the DAE-index is higher than 1
            
                :rtype: None
                """    

   Note that there are plenty of pydocstring-plugins around for different IDEs that autogenerate Python docstrings.

Generate the documentation, including API:

.. code-block:: bash

    $ cd docs/
    $ sphinx-apidoc -f -o source ../awebox
    $ make html

Attention: don't git commit the auto-generated `*.rst`-files or the `build/`-folder!

Inspect the API in `docs/build/html/modules.html`, and the project documentation in `docs/build/html/index.html`.

Useful links 
---------------

Some reading tips on (Python-based) software development.

- https://docs.python-guide.org A best practice handbook for Python.
- http://book.pythontips.com/en/latest/ : Python coding tips.
- https://twsba16.readthedocs.io/en/latest/ : TEMPO workshop on software development.
- https://www.joelonsoftware.com/ : Software development blog.

Some useful tools:

- https://pypi.org/project/ipdb/ : interactive Python debugger for the terminal