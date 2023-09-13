
 The basic type used in luac 5.4 (// lua5.4.1/src/lundump.c):

 ``` text
 Varint : Variable length integer

 Vector {
     Varint len;
     T      elem[len];
 }

 String {
     Varint len;
     
     union {
         _   null  (if len == 0)
         u8  str[len - 1]
     }
 }
 ```

 The Format of Lua 5.4 binary chunk is like:
 ``` text
 BinaryChunk {
     u8  lua_signature[4];         -> "0x1bLua"  (<esc>Lua)
     
     u8  luac_version;             -> 0x54
     u8  luac_format;              -> 0 : official luac output format
     u8  luac_magic[6];            -> "\x19\x93\r\n\x1a\n",  magic number
     u8  instruction_size;         -> 4
     u8  lua_int_size;             -> 8  size of isize
     u8  lua_float_size;           -> 8  size of f64
     u8  luac_int[8];              -> 0x5678
     u8  luac_num[8];              -> 370.5
                 
     Prototype main_func;     
 }

 ProtoType {
     u8          upvalue_size;
     String      source;
     varint      line_defined;
     varint      last_line_defined;
     byte        num_params;
     byte        is_vararg;
     byte        max_stack_size;
     varint      size_code;
     Instruction code[size_code];
     varint      size_constants;
     Constant    constants[size_constants];
     varint      size_upvalues;
     Upvalue     upvalues[size_upvalues];
     varint      size_protos;
     Prototype   protos[size_protos];            
     varint      size_line_info;
     byte        line_info[size_line_info];
     varint      size_abs_line_info;
     AbsLineInfo abs_line_info[size_abs_line_info];
     varint      size_loc_vars;
     LocVar      loc_vars[size_loc_vars];
     varint      size_upval_names;
     String      upval_names[size_upval_names];
 }


 Constant {
    byte tag;
    union {
      _           nil_value;
      _           false_alue;
      _           true_value;
      f64         float_value;
      i64         int_value;
      String      str_value;
    } value;
  }

 Constant Tag and Value:
 Tag     Value
 0x00	nil
 0x01	false
 0x11	true
 0x03	i64
 0x13	f64
 0x04	short string (String)
 0x14	long string (String)


 Upvalue {
   byte instack;
   byte idx;
   byte kind;
 }

 LocVar {
   String name;
   varint start_pc;
   varint end_pc;
 }


 AbsLineInfo {
     varint pc;
     varint line;  
 }

 ```