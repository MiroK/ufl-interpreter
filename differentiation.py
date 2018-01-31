import numpy as np
import dolfin
import iufl


def eval_grad_foo(foo):
    '''Exact derivatives of function'''

    def evaluate(x, foo=foo):
        x = np.fromiter(x, dtype=float)

        V = foo.function_space()
        el = V.element()
        mesh = V.mesh()
        # Find the cell with point
        x_point = dolfin.Point(*x) 
        cell_id = mesh.bounding_box_tree().compute_first_entity_collision(x_point)
        cell = dolfin.Cell(mesh, cell_id)
        coordinate_dofs = cell.get_vertex_coordinates()

        shape = dolfin.grad(foo).ufl_shape
        # grad grad grad ... ntime is stored in
        nslots = np.prod(shape)

        # Array for values with derivatives of all basis functions. 4 * element dim
        values = np.zeros(nslots*el.space_dimension(), dtype=float)
        # Compute all 2nd order derivatives
        el.evaluate_basis_derivatives_all(1, values, x, coordinate_dofs, cell.orientation())
        # Reshape such that colums are [d/dxx, d/dxy, d/dyx, d/dyy]
        values = values.reshape((-1, nslots))

        # Get expansion coefs on cell. Alternative to this is f.restrict(...)
        dofs = V.dofmap().cell_dofs(cell_id)
        dofs = foo.vector()[dofs]

        # Perform reduction on each colum - you get that deriv of foo
        values = np.array([np.inner(row, dofs) for row in values.T])
        values = values.reshape(shape)

        return values
    return evaluate


def eval_grad_expr(foo, mesh):
    '''Derivatives of polyn fit of foo'''
    assert mesh is not None
    # The idea is to compute df  as sum c_k dphi_k    
    def evaluate(x, foo=foo, mesh=mesh):
        x = np.fromiter(x, dtype=float)

        ufl_element = iufl.get_element(foo)
        # NOTE: this is here to make evaluating dofs simpler. Too lazy
        # now to handle Hdiv and what not
        assert ufl_element.family() in ('Lagrange', 'Discontinuous Lagrange')

        V = dolfin.FunctionSpace(mesh, ufl_element)
        
        el = V.element()
        mesh = V.mesh()
        # Find the cell with point
        x_point = dolfin.Point(*x) 
        cell_id = mesh.bounding_box_tree().compute_first_entity_collision(x_point)
        cell = dolfin.Cell(mesh, cell_id)
        coordinate_dofs = cell.get_vertex_coordinates()

        shape = dolfin.grad(foo).ufl_shape
        # grad grad grad ... ntime is stored in
        nslots = np.prod(shape)

        # Array for values with derivatives of all basis functions. 4 * element dim
        values = np.zeros(nslots*el.space_dimension(), dtype=float)
        # Compute all 2nd order derivatives
        el.evaluate_basis_derivatives_all(1, values, x, coordinate_dofs, cell.orientation())
        # Reshape such that colums are [d/dxx, d/dxy, d/dyx, d/dyy]
        values = values.reshape((-1, nslots))

        # Get expansion coefs on cell.
        indices = list(V.dofmap().cell_dofs(cell_id))
        dofs_x = V.tabulate_dof_coordinates().reshape((V.dim(), -1))[indices]

        # Sclar spaces
        if V.num_sub_spaces() == 0:
            dofs = np.array([foo(xj) for xj in dofs_x])
        # Not scalar spaces are filled by components
        else:
            dofs = np.zeros(len(indices), dtype=float)

            for comp in range(V.num_sub_spaces()):
                # Global
                comp_indices = list(V.sub(comp).dofmap().cell_dofs(cell_id))
                # Local w.r.t to all dofs
                local = [indices.index(comp_id) for comp_id in comp_indices]
                for lid in local:
                    dofs[lid] = foo(dofs_x[lid])[comp]

        # Perform reduction on each colum - you get that deriv of foo
        values = np.array([np.inner(row, dofs) for row in values.T])
        values = values.reshape(shape)

        return values
    return evaluate    
