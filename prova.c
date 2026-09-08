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


// iif statement

"iif" {return IIF;}



%token <ifStmt> IIF;

statement
    :
    | iff_statement

iif_statement
    : IIF LPAR assign_statement SEMI exp RPAR
    {
        createLabel(program, $1.lElse);
        createLabel(program, $1.lExit);
        BEQ(program, $5, REG_0, $1.lElse);
    }
    code_block
    {
        // if computation
        genJ(program, $1.lExit);
        assignLabel(program, $1.lElse);
    }
    else_part
    {
        // else computation
        assignLabel(program, $1.lExit)
    }

 // count_while
 
"count_while" { return COUNT_WHILE; }

// In parser.y (dichiarazioni)
typedef struct {
    t_label *lLoop;
    t_label *lExit;
    t_regID rCount; // Uniformato a rCount
} t_count_while_stmt;

%union {
    // ...
    t_count_while_stmt cwhileStmt; // Nome del tipo corretto
}

%token <cwhileStmt> COUNT_WHILE

// Nelle regole
statement
    // ...
    | count_while_statement SEMI // Aggiunto SEMI
    ;

count_while_statement
    : var_id ASSIGN COUNT_WHILE
    {
        // Controllo richiesto dalla traccia: deve essere uno scalare
        if (isArray($1)) {
            yyerror("La destinazione deve essere una variabile scalare");
            YYERROR;
        }

        $3.lLoop = createLabel(program);
        $3.lExit = createLabel(program);
        
        $3.rCount = getNewRegister(program);
        genLI(program, $3.rCount, 0);
        
        // Etichetta per il ricalcolo della condizione
        assignLabel(program, $3.lLoop);
    }
    LPAR exp RPAR
    {
        // (Rimossa la doppia assignLabel)
        // Se la condizione è falsa (0), esco dal ciclo
        genBEQ(program, $6, REG_0, $3.lExit);
    }
    code_block
    {
        // 1. Incremento il contatore (genADDI corretto)
        genADDI(program, $3.rCount, $3.rCount, 1);
        
        // 2. Torno all'inizio per rivalutare la condizione
        genJ(program, $3.lLoop);
        
        // 3. Fine del ciclo
        assignLabel(program, $3.lExit);
        
        // 4. SOLO ORA salvo il contatore nella variabile, fuori dal loop!
        genStoreRegisterToVariable(program, $1, $3.rCount);
    }
;

// if repeat

"if_repeat" { return IFR; }
"until"     { return UN; }

// Nota: tolto il punto e virgola dalla definizione del token
%token <whileStmt> IFR
%token UN 

statement
    // ...
    | IFR LPAR exp RPAR
    {
        $1.lLoop = createLabel(program);
        $1.lExit = createLabel(program);

        // 1. Se exp1 è falsa (0), esco subito scavalcando tutto
        genBEQ(program, $3, REG_0, $1.lExit);

        // 2. Metto QUI l'etichetta di loop, così il code_block che 
        // verrà generato subito dopo si troverà sotto l'etichetta!
        assignLabel(program, $1.lLoop);
    }
    code_block UN LPAR exp RPAR
    {
        // 3. Logica UNTIL: se exp2 (ora in $9) è FALSA (0), torno su a ripetere.
        // Se è vera, non salto e cado morbidamente nell'etichetta di uscita.
        genBEQ(program, $9, REG_0, $1.lLoop);
        
        // 4. Etichetta di uscita finale
        assignLabel(program, $1.lExit);
    }
;

// repeat_exp

"if_repeat" {return IFR;}

typedef struct{
    l_label *lLoop;
    l_label *lExit;
    t_regID rRes;
} ifr_stmt;

%union {
    ifr_stmt ifrStmt;
}

%token <ifrStmt> IFR;

statement
    :
    | REXP RPAR var_id ASSIGN exp COMMA exp
    {
        $1.lLoop = createLabel(program);
        $1.lExit = createLabel(program);

        $1.rRes = genNewRegister(program);
        genAdd(program, $1.rRes, REG_0, $5);
        genStoreToVariable(program, $3, $1.rRes);

        assignLabel(program, $1.lLoop);
        BLE(program, $7, REG_0, $1.lExit);
    }
    COMMA exp RPAR SEMICOL
    {
        // esecuzione codice
        genAdd(program, $1.rRes, REG_0, $10);
        genStoreToVariable(program, $3, $1.rRes);
        genJ(program, $1.lLoop);

        assignLabel(program, $1.lExit);
    }
;    

// converge

statement
    :
    | CONV var_id
    {
        $1.lLoop = createLabel(program);
        $1.lExit = createLabel(program);

        assignLabel($1.lLoop);
    }
    code_block
    {
        assignLabel($1.lLoop);
    }


// zip

// 1. controllo dimensioni, conto dimensione piu piccola, inizializzo registro dim_min
// 2. inserisco in output solo fino a grandezza array3 -> altro registro

statement
    :
    | ZIP var_id COMMA var_id COMMA var_id
    {
        if(!isArray($2) || !isArray($4) || !isArray($6)){
            yyerror("errore");
            YYERROR;
        }

        $1.lLoop = createLabel(program);
        $1.lExit = createLabel(program);
        $1.rIdx = genNewRegister(program);
        genLI(program, $1.rIdx = )

        t_regID rMin = genNewRegister(program);
        t_regID rMin = genNewRegister(program);

        int len1 = $1->arraySize();
        int len2 = $3->arraySize();

        if(len1 < len2){
            genLI(program, rMin, len1);
        } else {
            genLI(program, rMin, len2);
        }
        $1.genLI(program, );
        
    }


list
    | exp list
    {

    }
    | exp
    {

    }

statement
    :
    | var_id ASSIGN ALL LPAR list RPAR
    {
        
    }   



        





        

