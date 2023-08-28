
-- local num = 99

-- tab = { a = {print = print}}

-- tab.a.print "hello world"

-- function add(a, b)
--   return a + b
-- end

local x, y = print "hello world"

print "hello world"


-- local y = x
-- y = y + 1
-- return x, y

--[=[
    main <.\testes\_.lua:0,0> (5 instructions at 010AEAE0)       
    0+ params, 2 slots, 1 upvalue, 0 locals, 2 constants, 0 functions
            1       [1]     VARARGPREP      0
            2       [12]    GETTABUP        0 0 0   ; _ENV "print"
            3       [12]    LOADK           1 1     ; "hello world"
            4       [12]    CALL            0 2 1   ; 1 in 0 out
            5       [12]    RETURN          0 1 1   ; 0 out
    constants (2) for 010AEAE0:
            0       S       "print"
            1       S       "hello world"
    locals (0) for 010AEAE0:
    upvalues (1) for 010AEAE0:
            0       _ENV    1       0
]=]