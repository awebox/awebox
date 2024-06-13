import awebox.tools.constraint_operations as cstr_op
import awebox.tools.debug_operations as debug_op
import awebox.tools.performance_operations as perf_op
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op

def test_cstr_op():
    cstr_op.test()

def test_debug_op():
    debug_op.test()

def test_perf_op():
    perf_op.test()
    
def test_print_op():
    print_op.test()

def test_save_op():
    save_op.test()

def test_struct_op():
    struct_op.test()

def test_vect_op():
    vect_op.test()


if __name__ == "__main__":
    test_cstr_op()
    test_debug_op()
    test_perf_op()
    test_print_op()
    test_save_op()
    test_struct_op()
    test_vect_op()