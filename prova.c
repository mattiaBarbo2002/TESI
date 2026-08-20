yyerror("error");
YYERROR;

typedef struct {
    t_label *lLoop;
    t_label *lExit;
    t_regID altro;
} t_nameStmt;


// 2015 fattoriale e absolute value


exp
    :
    | exp NOT_OP
    {
        $$ = getNewRegister(program)
        genLI(program, $$, 0)

        t_label* lZero = createLabel(program)
        t_label* lLoop = createLabel(program)
        t_regID rTemp = genNewRegister(program)
        genADDI(program, rTemp, $1, -1)

        BLE(program, $1, REG_0, lZero)
        
        assignLabel(program, lLoop)
        genMUL(program, $1, $1, rTemp)
        genADDI(program, rTemp, rTemp, -1)
        BGT(program, rTemp, REG_0, lLoop)

        genADD(program, $$, $$, $1)
        assignLabel(program, lZero)
    }
    | OR_OP exp OR_OP
    {
        $$ = getNewRegister(program)
        genLI(program, $$, 0)
        
        t_label* lPositive = createLabel(program)
        t_label* lExit = createLabel(program)
        l_label* lLoop = createLabel(program)
        t_regID rRes = getNewRegister
        genLI(program, rTemp, 0)

        BGE(progam, $2, REG_0, lPositive)

        assignLabel(program, lLoop)
        genADDI(program, $2, $2, 1)
        genADDI(program, rRes, rRes, 1)
        genBLT(program, $2, REG_0, lLoop)

        genADD(program, $$, $$, rRes)
        genJ(program, lExit)


        assignLabel(program, lPositive)
        genADD(program, $$, $$, $2)

        assignLabel(program, lExit)
    }

// 2023/06/30 divisione

exp
    :
    | var_id LSQUARE DIV_OP RSQUARE var_id
    {
        $$ = getNewRegister(program)
        genLI(program, $$, 0)

        t_rOne = getNewRegister(program)
        t_rTwo = getNewRegister(program)

        t_rOne = genLoadVariable(program, $1)
        t_rTwo = genLoadVariable(program, $5)

        t_label* lLoop = createLabel(program)
        t_label* lZero = createLabel(program)
        t_label* lExit = createLabel(program)

        genBEQ(program, t_rTwo, REG_0, lZero)
        
        assignLabel(program, lLoop)
        genBLT(program, t_rOne, t_rTwo, lExit)
        genADDI(program, $$, $$, 1)
        genSUB(program, t_rOne, t_rOne, t_rTwo)        
        genL(program, lLoop)

        assignLabel(program, lZero)
        genADDI(program, $$, $$, INT_MAX)

        assignLabel(program, lExit)      
    }

// exp 2024-07-04 tir

exp
    :
    | TRI LPAR exp RPAR
    {
        $$ = getNewRegister(program)
        genLI(program, $$, 0)

        t_label* lExit = createLabel(program)
        
        genBLE(program, $3, REG_0, lExit)

        genADDI(program, $$, $3, 1)
        genMUL(program, $$, $$, $3)
        genDEVI(program, $$, $$, 2) 

        assignLabel(program, lExit)
    }

// exp 2025 17 01

%token QUE COL
%left COL

exp
    :
    | exp QUE exp COL exp
    {
        $$ = getNewRegister(program);
        genLI(program, $$, 0);
    
        t_label* lExit = createLabel(program);
        t_label* lFalse = createLabel(program);

        BEQ(program, $1, REG_0, lFalse);

        genADD(program, $$, $$, $3);
        genJ(program, lExit);

        assignLabel(program, lFalse);
        genADD(program, $$, $$, $5);

        assignLabel(program, lExit);
    }

// 2025 06 05

exp
    :
    | ALL LPAR var_id COMMA var_id COMMA exp
    {
        $$ = getNewRegister(program)

        if(!isArray($3)){
            yyerror("non è un array");
            YYERROR;
        }

        t_label*
    } 

    // exp 2025 06 05

typedef struct{
    t_label* lLoop;
    t_label* lExit;
    t_regID rIdx;
    t_regID rRes;

} t_allExp;

t_allExp allExp;

%token <allExp> ALL

exp
    :
    | ALL LPAR var_id COMMA var_id COMMA 
    {
        $$ = getNewRegister(program);

        if(!isArray($3) || isArray($5)){
            yyerror("errore");
            YYERROR;
        }

        $1.lLoop = createLabel(program);
        $1.lExit = createLabel(program);

        $1.rIdx = getNewregister(program);
        genLI(program, $1.rIdx, 0);

        $1.rRes = getNewRegister(program);
        genLI(program, $1.rRes, 1);

        t_regID rSize = getNewRegister(program);
        genLI(program, rSize, $3->arraySize);
        
        // loop
        assignLabel(program, $1.lLoop);
        BEQ(program, $1.rIdx, rSize, lExit);

        t_regID rElm = genLoadArrayElement(program, $3, $1.rIdx);
        genStoreRegisterToVariable(program, $5, rElm);

        genADDI(program, $1.rIdx, 1);
    }
    exp RPAR 
    {   
        BEQ(program, $8, $1.rRes, lLoop);
        genSUBI(program, $1.rRes, 1);

        assignLabel(program, $1.lExit);
        genADDI(program, $$, $1.rRes, 0);
    }

// 2026 02 16 binop    

"binop" {return BINOP;}

%token BINOP

exp 
    :
    | BINOP LT exp GT LPAR exp COMMA exp RPAR
    {
        $$ = getNewRegister(program);
        
        t_label* lTrue = createLabel(program);
        t_label* lFalse = createLabel(program);
        t_label* lExit = createLabel(program);

        t_regID rTrue = getNewRegister(program);
        genLI(program, rTrue, 1);

        genBEQ(program, $3, REG_0, lFalse);
        genBEQ(program, $3, rTrue, lTrue);

        // non 0 o 1 -> XOR
        genXOR(program, $$, $6, $8);
        genJ(program, lExit);

        // 0 -> AND
        assignLabel(program, lFalse);
        genAND(program, $$, $6, $8);
        genJ(program, lExit);

        // 1 -> OR
        assignLabel(program, lTrue);
        genOR(program, $$, $6, $8);
        
        assignLabel(program, lExit);
    }

// 2026 06 16 let

"let" {return LET;}
"$" {return DOLLAR;}

%type <reg> exp_list

# define DIM 10
t_regID letArray[DIM]
t_regID rIdx = getNewRegister(program)
genLI(program, rIdx, 0)

exp_list: exp COMMA exp_list
    {
        genStoreRegisterToArrayElement(program, letArray, rIdx, $1);

    }
    | exp 
    {
        
    }


exp 
    :
    | LET LPAR exp_list RPAR {

    }






        





        

