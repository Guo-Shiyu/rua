function test(a, b, c) 
    if a then
        if b then 
            -- print("a=", a)
            -- print("b=", b)
        else 
            -- print("a=", a)
        end 
    elseif c then 
        -- print("c=", c)
    end 

    if c then 
        if b then 
            -- print("c=", c)
            -- print("b=", b)
        end 
    elseif b then 
        if a then 
            -- print("b=", b)
            -- print("a=", a)
        end 
    else
        if  b then 
            -- print("b=", b)
        end 
    end 
end 

test(1, nil, nil)
test(nil, 1, 2)
test(nil, nil, nil)
test(nil, nil, 2)
