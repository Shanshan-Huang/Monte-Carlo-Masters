       £K"	  јЋt„Abrain.Event:2уc”VВi     4Nd	ћaЅЋt„A"х“
f
S/sPlaceholder*'
_output_shapes
:€€€€€€€€€P*
shape:€€€€€€€€€P*
dtype0
f
R/rPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
S_/s_Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€P*
shape:€€€€€€€€€P
ґ
8Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"P      *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
©
7Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *    *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Ђ
9Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЪЩЩ>*+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Х
GActor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal8Actor/eval_net/l1/kernel/Initializer/random_normal/shape*
_output_shapes

:P*

seed*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
seed2*
dtype0
Ч
6Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulGActor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal9Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
А
2Actor/eval_net/l1/kernel/Initializer/random_normalAdd6Actor/eval_net/l1/kernel/Initializer/random_normal/mul7Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
є
Actor/eval_net/l1/kernel
VariableV2*
shape
:P*
dtype0*
_output_shapes

:P*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container 
ц
Actor/eval_net/l1/kernel/AssignAssignActor/eval_net/l1/kernel2Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:P
Щ
Actor/eval_net/l1/kernel/readIdentityActor/eval_net/l1/kernel*
_output_shapes

:P*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
†
(Actor/eval_net/l1/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*
valueB*Ќћћ=*)
_class
loc:@Actor/eval_net/l1/bias
≠
Actor/eval_net/l1/bias
VariableV2*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:
в
Actor/eval_net/l1/bias/AssignAssignActor/eval_net/l1/bias(Actor/eval_net/l1/bias/Initializer/Const*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
П
Actor/eval_net/l1/bias/readIdentityActor/eval_net/l1/bias*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
_output_shapes
:
Ю
Actor/eval_net/l1/MatMulMatMulS/sActor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
§
Actor/eval_net/l1/BiasAddBiasAddActor/eval_net/l1/MatMulActor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
k
Actor/eval_net/l1/ReluReluActor/eval_net/l1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
Є
9Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
:
Ђ
8Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
≠
:Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЪЩЩ>*,
_class"
 loc:@Actor/eval_net/a/a/kernel
Ш
HActor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*

seed*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
seed2*
dtype0*
_output_shapes

:
Ы
7Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulHActor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal:Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
Д
3Actor/eval_net/a/a/kernel/Initializer/random_normalAdd7Actor/eval_net/a/a/kernel/Initializer/random_normal/mul8Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
ї
Actor/eval_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:
ъ
 Actor/eval_net/a/a/kernel/AssignAssignActor/eval_net/a/a/kernel3Actor/eval_net/a/a/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(
Ь
Actor/eval_net/a/a/kernel/readIdentityActor/eval_net/a/a/kernel*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:*
T0
Ґ
)Actor/eval_net/a/a/bias/Initializer/ConstConst*
valueB*Ќћћ=**
_class 
loc:@Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
ѓ
Actor/eval_net/a/a/bias
VariableV2**
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ж
Actor/eval_net/a/a/bias/AssignAssignActor/eval_net/a/a/bias)Actor/eval_net/a/a/bias/Initializer/Const*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(
Т
Actor/eval_net/a/a/bias/readIdentityActor/eval_net/a/a/bias*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
:
≥
Actor/eval_net/a/a/MatMulMatMulActor/eval_net/l1/ReluActor/eval_net/a/a/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
І
Actor/eval_net/a/a/BiasAddBiasAddActor/eval_net/a/a/MatMulActor/eval_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
m
Actor/eval_net/a/a/TanhTanhActor/eval_net/a/a/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
`
Actor/eval_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
И
Actor/eval_net/a/scaled_aMulActor/eval_net/a/a/TanhActor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
Ї
:Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"P      *-
_class#
!loc:@Actor/target_net/l1/kernel*
dtype0*
_output_shapes
:
≠
9Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *    *-
_class#
!loc:@Actor/target_net/l1/kernel*
dtype0*
_output_shapes
: 
ѓ
;Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *ЪЩЩ>*-
_class#
!loc:@Actor/target_net/l1/kernel*
dtype0
Ы
IActor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:Actor/target_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:P*

seed*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
seed2(
Я
8Actor/target_net/l1/kernel/Initializer/random_normal/mulMulIActor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
_output_shapes

:P
И
4Actor/target_net/l1/kernel/Initializer/random_normalAdd8Actor/target_net/l1/kernel/Initializer/random_normal/mul9Actor/target_net/l1/kernel/Initializer/random_normal/mean*
_output_shapes

:P*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel
љ
Actor/target_net/l1/kernel
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *-
_class#
!loc:@Actor/target_net/l1/kernel*
	container *
shape
:P
ю
!Actor/target_net/l1/kernel/AssignAssignActor/target_net/l1/kernel4Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:P
Я
Actor/target_net/l1/kernel/readIdentityActor/target_net/l1/kernel*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
_output_shapes

:P
§
*Actor/target_net/l1/bias/Initializer/ConstConst*
_output_shapes
:*
valueB*Ќћћ=*+
_class!
loc:@Actor/target_net/l1/bias*
dtype0
±
Actor/target_net/l1/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@Actor/target_net/l1/bias
к
Actor/target_net/l1/bias/AssignAssignActor/target_net/l1/bias*Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
Х
Actor/target_net/l1/bias/readIdentityActor/target_net/l1/bias*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
_output_shapes
:
§
Actor/target_net/l1/MatMulMatMulS_/s_Actor/target_net/l1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
™
Actor/target_net/l1/BiasAddBiasAddActor/target_net/l1/MatMulActor/target_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
o
Actor/target_net/l1/ReluReluActor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Љ
;Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"      *.
_class$
" loc:@Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
:
ѓ
:Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
±
<Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЪЩЩ>*.
_class$
" loc:@Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
Ю
JActor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
seed28
£
9Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulJActor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:
М
5Actor/target_net/a/a/kernel/Initializer/random_normalAdd9Actor/target_net/a/a/kernel/Initializer/random_normal/mul:Actor/target_net/a/a/kernel/Initializer/random_normal/mean*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:*
T0
њ
Actor/target_net/a/a/kernel
VariableV2*
_output_shapes

:*
shared_name *.
_class$
" loc:@Actor/target_net/a/a/kernel*
	container *
shape
:*
dtype0
В
"Actor/target_net/a/a/kernel/AssignAssignActor/target_net/a/a/kernel5Actor/target_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Ґ
 Actor/target_net/a/a/kernel/readIdentityActor/target_net/a/a/kernel*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:
¶
+Actor/target_net/a/a/bias/Initializer/ConstConst*
valueB*Ќћћ=*,
_class"
 loc:@Actor/target_net/a/a/bias*
dtype0*
_output_shapes
:
≥
Actor/target_net/a/a/bias
VariableV2*
shared_name *,
_class"
 loc:@Actor/target_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
о
 Actor/target_net/a/a/bias/AssignAssignActor/target_net/a/a/bias+Actor/target_net/a/a/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias
Ш
Actor/target_net/a/a/bias/readIdentityActor/target_net/a/a/bias*
_output_shapes
:*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias
є
Actor/target_net/a/a/MatMulMatMulActor/target_net/l1/Relu Actor/target_net/a/a/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
≠
Actor/target_net/a/a/BiasAddBiasAddActor/target_net/a/a/MatMulActor/target_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
q
Actor/target_net/a/a/TanhTanhActor/target_net/a/a/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
b
Actor/target_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
О
Actor/target_net/a/scaled_aMulActor/target_net/a/a/TanhActor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
J
mul/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
[
mulMulmul/xActor/target_net/l1/kernel/read*
T0*
_output_shapes

:P
L
mul_1/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
]
mul_1Mulmul_1/xActor/eval_net/l1/kernel/read*
T0*
_output_shapes

:P
?
addAddmulmul_1*
_output_shapes

:P*
T0
≤
AssignAssignActor/target_net/l1/kerneladd*-
_class#
!loc:@Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0
L
mul_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
Y
mul_2Mulmul_2/xActor/target_net/l1/bias/read*
_output_shapes
:*
T0
L
mul_3/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
W
mul_3Mulmul_3/xActor/eval_net/l1/bias/read*
T0*
_output_shapes
:
?
add_1Addmul_2mul_3*
_output_shapes
:*
T0
Ѓ
Assign_1AssignActor/target_net/l1/biasadd_1*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
L
mul_4/xConst*
_output_shapes
: *
valueB
 *§p}?*
dtype0
`
mul_4Mulmul_4/x Actor/target_net/a/a/kernel/read*
T0*
_output_shapes

:
L
mul_5/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
^
mul_5Mulmul_5/xActor/eval_net/a/a/kernel/read*
T0*
_output_shapes

:
C
add_2Addmul_4mul_5*
T0*
_output_shapes

:
Є
Assign_2AssignActor/target_net/a/a/kerneladd_2*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
L
mul_6/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
Z
mul_6Mulmul_6/xActor/target_net/a/a/bias/read*
_output_shapes
:*
T0
L
mul_7/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
X
mul_7Mulmul_7/xActor/eval_net/a/a/bias/read*
T0*
_output_shapes
:
?
add_3Addmul_6mul_7*
_output_shapes
:*
T0
∞
Assign_3AssignActor/target_net/a/a/biasadd_3*
use_locking(*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
p
Critic/StopGradientStopGradientActor/eval_net/a/scaled_a*
T0*'
_output_shapes
:€€€€€€€€€
і
7Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"P      **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
І
6Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
©
8Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *Ќћћ=**
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
Т
FCritic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
seed2c*
dtype0*
_output_shapes

:P*

seed*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
У
5Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulFCritic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal8Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
ь
1Critic/eval_net/l1/w1_s/Initializer/random_normalAdd5Critic/eval_net/l1/w1_s/Initializer/random_normal/mul6Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes

:P*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
Ј
Critic/eval_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes

:P*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P
т
Critic/eval_net/l1/w1_s/AssignAssignCritic/eval_net/l1/w1_s1Critic/eval_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
Ц
Critic/eval_net/l1/w1_s/readIdentityCritic/eval_net/l1/w1_s*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
і
7Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"      **
_class 
loc:@Critic/eval_net/l1/w1_a*
dtype0
І
6Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    **
_class 
loc:@Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
©
8Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *Ќћћ=**
_class 
loc:@Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
Т
FCritic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
seed2l*
dtype0*
_output_shapes

:*

seed
У
5Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulFCritic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal8Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
ь
1Critic/eval_net/l1/w1_a/Initializer/random_normalAdd5Critic/eval_net/l1/w1_a/Initializer/random_normal/mul6Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
Ј
Critic/eval_net/l1/w1_a
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a
т
Critic/eval_net/l1/w1_a/AssignAssignCritic/eval_net/l1/w1_a1Critic/eval_net/l1/w1_a/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
Ц
Critic/eval_net/l1/w1_a/readIdentityCritic/eval_net/l1/w1_a*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
¶
'Critic/eval_net/l1/b1/Initializer/ConstConst*
valueB*Ќћћ=*(
_class
loc:@Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
≥
Critic/eval_net/l1/b1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1
в
Critic/eval_net/l1/b1/AssignAssignCritic/eval_net/l1/b1'Critic/eval_net/l1/b1/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1
Р
Critic/eval_net/l1/b1/readIdentityCritic/eval_net/l1/b1*
_output_shapes

:*
T0*(
_class
loc:@Critic/eval_net/l1/b1
Ю
Critic/eval_net/l1/MatMulMatMulS/sCritic/eval_net/l1/w1_s/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
∞
Critic/eval_net/l1/MatMul_1MatMulCritic/StopGradientCritic/eval_net/l1/w1_a/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
З
Critic/eval_net/l1/addAddCritic/eval_net/l1/MatMulCritic/eval_net/l1/MatMul_1*'
_output_shapes
:€€€€€€€€€*
T0
Е
Critic/eval_net/l1/add_1AddCritic/eval_net/l1/addCritic/eval_net/l1/b1/read*
T0*'
_output_shapes
:€€€€€€€€€
k
Critic/eval_net/l1/ReluReluCritic/eval_net/l1/add_1*'
_output_shapes
:€€€€€€€€€*
T0
¬
>Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
:
µ
=Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
dtype0
Ј
?Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
І
MCritic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal>Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
seed2~*
dtype0*
_output_shapes

:*

seed
ѓ
<Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulMCritic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormal?Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
Ш
8Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd<Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul=Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
≈
Critic/eval_net/q/dense/kernel
VariableV2*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
О
%Critic/eval_net/q/dense/kernel/AssignAssignCritic/eval_net/q/dense/kernel8Critic/eval_net/q/dense/kernel/Initializer/random_normal*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
Ђ
#Critic/eval_net/q/dense/kernel/readIdentityCritic/eval_net/q/dense/kernel*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
ђ
.Critic/eval_net/q/dense/bias/Initializer/ConstConst*
valueB*Ќћћ=*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
є
Critic/eval_net/q/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:
ъ
#Critic/eval_net/q/dense/bias/AssignAssignCritic/eval_net/q/dense/bias.Critic/eval_net/q/dense/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias
°
!Critic/eval_net/q/dense/bias/readIdentityCritic/eval_net/q/dense/bias*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
Њ
Critic/eval_net/q/dense/MatMulMatMulCritic/eval_net/l1/Relu#Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
ґ
Critic/eval_net/q/dense/BiasAddBiasAddCritic/eval_net/q/dense/MatMul!Critic/eval_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
Є
9Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"P      *,
_class"
 loc:@Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
:
Ђ
8Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@Critic/target_net/l1/w1_s*
dtype0
≠
:Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *Ќћћ=*,
_class"
 loc:@Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
Щ
HCritic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
seed2Н*
dtype0*
_output_shapes

:P*

seed*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s
Ы
7Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulHCritic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P
Д
3Critic/target_net/l1/w1_s/Initializer/random_normalAdd7Critic/target_net/l1/w1_s/Initializer/random_normal/mul8Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P
ї
Critic/target_net/l1/w1_s
VariableV2*,
_class"
 loc:@Critic/target_net/l1/w1_s*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name 
ъ
 Critic/target_net/l1/w1_s/AssignAssignCritic/target_net/l1/w1_s3Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
Ь
Critic/target_net/l1/w1_s/readIdentityCritic/target_net/l1/w1_s*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P
Є
9Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
:
Ђ
8Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
≠
:Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *Ќћћ=*,
_class"
 loc:@Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
Щ
HCritic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
seed2Ц
Ы
7Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulHCritic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
_output_shapes

:
Д
3Critic/target_net/l1/w1_a/Initializer/random_normalAdd7Critic/target_net/l1/w1_a/Initializer/random_normal/mul8Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
_output_shapes

:*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a
ї
Critic/target_net/l1/w1_a
VariableV2*,
_class"
 loc:@Critic/target_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
ъ
 Critic/target_net/l1/w1_a/AssignAssignCritic/target_net/l1/w1_a3Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ь
Critic/target_net/l1/w1_a/readIdentityCritic/target_net/l1/w1_a*,
_class"
 loc:@Critic/target_net/l1/w1_a*
_output_shapes

:*
T0
™
)Critic/target_net/l1/b1/Initializer/ConstConst*
_output_shapes

:*
valueB*Ќћћ=**
_class 
loc:@Critic/target_net/l1/b1*
dtype0
Ј
Critic/target_net/l1/b1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/target_net/l1/b1
к
Critic/target_net/l1/b1/AssignAssignCritic/target_net/l1/b1)Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
Ц
Critic/target_net/l1/b1/readIdentityCritic/target_net/l1/b1*
T0**
_class 
loc:@Critic/target_net/l1/b1*
_output_shapes

:
§
Critic/target_net/l1/MatMulMatMulS_/s_Critic/target_net/l1/w1_s/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Љ
Critic/target_net/l1/MatMul_1MatMulActor/target_net/a/scaled_aCritic/target_net/l1/w1_a/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Н
Critic/target_net/l1/addAddCritic/target_net/l1/MatMulCritic/target_net/l1/MatMul_1*
T0*'
_output_shapes
:€€€€€€€€€
Л
Critic/target_net/l1/add_1AddCritic/target_net/l1/addCritic/target_net/l1/b1/read*'
_output_shapes
:€€€€€€€€€*
T0
o
Critic/target_net/l1/ReluReluCritic/target_net/l1/add_1*'
_output_shapes
:€€€€€€€€€*
T0
∆
@Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"      *3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
dtype0
є
?Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
ї
ACritic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *Ќћћ=*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
Ѓ
OCritic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
seed2®*
dtype0*
_output_shapes

:*

seed
Ј
>Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulOCritic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalACritic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
_output_shapes

:
†
:Critic/target_net/q/dense/kernel/Initializer/random_normalAdd>Critic/target_net/q/dense/kernel/Initializer/random_normal/mul?Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel
…
 Critic/target_net/q/dense/kernel
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
	container 
Ц
'Critic/target_net/q/dense/kernel/AssignAssign Critic/target_net/q/dense/kernel:Critic/target_net/q/dense/kernel/Initializer/random_normal*
use_locking(*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
±
%Critic/target_net/q/dense/kernel/readIdentity Critic/target_net/q/dense/kernel*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
∞
0Critic/target_net/q/dense/bias/Initializer/ConstConst*
valueB*Ќћћ=*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
dtype0*
_output_shapes
:
љ
Critic/target_net/q/dense/bias
VariableV2*
shared_name *1
_class'
%#loc:@Critic/target_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
В
%Critic/target_net/q/dense/bias/AssignAssignCritic/target_net/q/dense/bias0Critic/target_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
І
#Critic/target_net/q/dense/bias/readIdentityCritic/target_net/q/dense/bias*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
_output_shapes
:
ƒ
 Critic/target_net/q/dense/MatMulMatMulCritic/target_net/l1/Relu%Critic/target_net/q/dense/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Љ
!Critic/target_net/q/dense/BiasAddBiasAdd Critic/target_net/q/dense/MatMul#Critic/target_net/q/dense/bias/read*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
S
target_q/mul/xConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
x
target_q/mulMultarget_q/mul/x!Critic/target_net/q/dense/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
X
target_q/addAddR/rtarget_q/mul*'
_output_shapes
:€€€€€€€€€*
T0
Р
TD_error/SquaredDifferenceSquaredDifferencetarget_q/addCritic/eval_net/q/dense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
_
TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

TD_error/MeanMeanTD_error/SquaredDifferenceTD_error/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Z
C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
`
C_train/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  А?*
dtype0
З
C_train/gradients/FillFillC_train/gradients/ShapeC_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Г
2C_train/gradients/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ї
,C_train/gradients/TD_error/Mean_grad/ReshapeReshapeC_train/gradients/Fill2C_train/gradients/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
Д
*C_train/gradients/TD_error/Mean_grad/ShapeShapeTD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
ѕ
)C_train/gradients/TD_error/Mean_grad/TileTile,C_train/gradients/TD_error/Mean_grad/Reshape*C_train/gradients/TD_error/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
Ж
,C_train/gradients/TD_error/Mean_grad/Shape_1ShapeTD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
o
,C_train/gradients/TD_error/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
t
*C_train/gradients/TD_error/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
…
)C_train/gradients/TD_error/Mean_grad/ProdProd,C_train/gradients/TD_error/Mean_grad/Shape_1*C_train/gradients/TD_error/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
v
,C_train/gradients/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ќ
+C_train/gradients/TD_error/Mean_grad/Prod_1Prod,C_train/gradients/TD_error/Mean_grad/Shape_2,C_train/gradients/TD_error/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
.C_train/gradients/TD_error/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
µ
,C_train/gradients/TD_error/Mean_grad/MaximumMaximum+C_train/gradients/TD_error/Mean_grad/Prod_1.C_train/gradients/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
≥
-C_train/gradients/TD_error/Mean_grad/floordivFloorDiv)C_train/gradients/TD_error/Mean_grad/Prod,C_train/gradients/TD_error/Mean_grad/Maximum*
_output_shapes
: *
T0
†
)C_train/gradients/TD_error/Mean_grad/CastCast-C_train/gradients/TD_error/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0*
Truncate( 
њ
,C_train/gradients/TD_error/Mean_grad/truedivRealDiv)C_train/gradients/TD_error/Mean_grad/Tile)C_train/gradients/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
Г
7C_train/gradients/TD_error/SquaredDifference_grad/ShapeShapetarget_q/add*
_output_shapes
:*
T0*
out_type0
Ш
9C_train/gradients/TD_error/SquaredDifference_grad/Shape_1ShapeCritic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
С
GC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs7C_train/gradients/TD_error/SquaredDifference_grad/Shape9C_train/gradients/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ђ
8C_train/gradients/TD_error/SquaredDifference_grad/scalarConst-^C_train/gradients/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
÷
5C_train/gradients/TD_error/SquaredDifference_grad/MulMul8C_train/gradients/TD_error/SquaredDifference_grad/scalar,C_train/gradients/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
ћ
5C_train/gradients/TD_error/SquaredDifference_grad/subSubtarget_q/addCritic/eval_net/q/dense/BiasAdd-^C_train/gradients/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
ё
7C_train/gradients/TD_error/SquaredDifference_grad/mul_1Mul5C_train/gradients/TD_error/SquaredDifference_grad/Mul5C_train/gradients/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:€€€€€€€€€
ю
5C_train/gradients/TD_error/SquaredDifference_grad/SumSum7C_train/gradients/TD_error/SquaredDifference_grad/mul_1GC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ф
9C_train/gradients/TD_error/SquaredDifference_grad/ReshapeReshape5C_train/gradients/TD_error/SquaredDifference_grad/Sum7C_train/gradients/TD_error/SquaredDifference_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
В
7C_train/gradients/TD_error/SquaredDifference_grad/Sum_1Sum7C_train/gradients/TD_error/SquaredDifference_grad/mul_1IC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ъ
;C_train/gradients/TD_error/SquaredDifference_grad/Reshape_1Reshape7C_train/gradients/TD_error/SquaredDifference_grad/Sum_19C_train/gradients/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Ђ
5C_train/gradients/TD_error/SquaredDifference_grad/NegNeg;C_train/gradients/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
Њ
BC_train/gradients/TD_error/SquaredDifference_grad/tuple/group_depsNoOp6^C_train/gradients/TD_error/SquaredDifference_grad/Neg:^C_train/gradients/TD_error/SquaredDifference_grad/Reshape
÷
JC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity9C_train/gradients/TD_error/SquaredDifference_grad/ReshapeC^C_train/gradients/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:€€€€€€€€€
–
LC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity5C_train/gradients/TD_error/SquaredDifference_grad/NegC^C_train/gradients/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*H
_class>
<:loc:@C_train/gradients/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€
џ
BC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradLC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
г
GC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpC^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradM^C_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1
п
OC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityLC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1H^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@C_train/gradients/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€
з
QC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityBC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradH^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*U
_classK
IGloc:@C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad
Ф
<C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulOC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency#Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
Б
>C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMulCritic/eval_net/l1/ReluOC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
ќ
FC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOp=^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul?^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
д
NC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity<C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulG^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*O
_classE
CAloc:@C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul
б
PC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1Identity>C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1G^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
ё
7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradReluGradNC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyCritic/eval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€
Л
5C_train/gradients/Critic/eval_net/l1/add_1_grad/ShapeShapeCritic/eval_net/l1/add*
out_type0*
_output_shapes
:*
T0
И
7C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Л
EC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs5C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape7C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ъ
3C_train/gradients/Critic/eval_net/l1/add_1_grad/SumSum7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradEC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
о
7C_train/gradients/Critic/eval_net/l1/add_1_grad/ReshapeReshape3C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum5C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ю
5C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum_1Sum7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradGC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
л
9C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape5C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum_17C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Њ
@C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp8^C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape:^C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1
ќ
HC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity7C_train/gradients/Critic/eval_net/l1/add_1_grad/ReshapeA^C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape
Ћ
JC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity9C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1A^C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:
М
3C_train/gradients/Critic/eval_net/l1/add_grad/ShapeShapeCritic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
Р
5C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1ShapeCritic/eval_net/l1/MatMul_1*
out_type0*
_output_shapes
:*
T0
Е
CC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs3C_train/gradients/Critic/eval_net/l1/add_grad/Shape5C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
З
1C_train/gradients/Critic/eval_net/l1/add_grad/SumSumHC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyCC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
и
5C_train/gradients/Critic/eval_net/l1/add_grad/ReshapeReshape1C_train/gradients/Critic/eval_net/l1/add_grad/Sum3C_train/gradients/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Л
3C_train/gradients/Critic/eval_net/l1/add_grad/Sum_1SumHC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyEC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
о
7C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1Reshape3C_train/gradients/Critic/eval_net/l1/add_grad/Sum_15C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Є
>C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp6^C_train/gradients/Critic/eval_net/l1/add_grad/Reshape8^C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1
∆
FC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity5C_train/gradients/Critic/eval_net/l1/add_grad/Reshape?^C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*H
_class>
<:loc:@C_train/gradients/Critic/eval_net/l1/add_grad/Reshape
ћ
HC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity7C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1?^C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
€
7C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMulMatMulFC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependencyCritic/eval_net/l1/w1_s/read*
T0*'
_output_shapes
:€€€€€€€€€P*
transpose_a( *
transpose_b(
я
9C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sFC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:P*
transpose_a(*
transpose_b( 
њ
AC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp8^C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul:^C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1
–
IC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity7C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMulB^C_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€P*
T0
Ќ
KC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity9C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1B^C_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:P
Г
9C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulHC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Critic/eval_net/l1/w1_a/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
у
;C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMulCritic/StopGradientHC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
≈
CC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp:^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul<^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
Ў
KC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity9C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulD^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:€€€€€€€€€
’
MC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity;C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1D^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*N
_classD
B@loc:@C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:*
T0
Р
!C_train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*(
_class
loc:@Critic/eval_net/l1/b1
°
C_train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@Critic/eval_net/l1/b1
–
C_train/beta1_power/AssignAssignC_train/beta1_power!C_train/beta1_power/initial_value*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(
Д
C_train/beta1_power/readIdentityC_train/beta1_power*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: *
T0
Р
!C_train/beta2_power/initial_valueConst*
valueB
 *wЊ?*(
_class
loc:@Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
°
C_train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape: 
–
C_train/beta2_power/AssignAssignC_train/beta2_power!C_train/beta2_power/initial_value*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
Д
C_train/beta2_power/readIdentityC_train/beta2_power*
_output_shapes
: *
T0*(
_class
loc:@Critic/eval_net/l1/b1
√
FC_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB"P      *
dtype0*
_output_shapes
:
≠
<C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: **
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB
 *    
Ђ
6C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillFC_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor<C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*
_output_shapes

:P*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*

index_type0
ƒ
$C_train/Critic/eval_net/l1/w1_s/Adam
VariableV2**
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name 
С
+C_train/Critic/eval_net/l1/w1_s/Adam/AssignAssign$C_train/Critic/eval_net/l1/w1_s/Adam6C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
∞
)C_train/Critic/eval_net/l1/w1_s/Adam/readIdentity$C_train/Critic/eval_net/l1/w1_s/Adam*
_output_shapes

:P*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
≈
HC_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB"P      *
dtype0*
_output_shapes
:
ѓ
>C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
±
8C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillHC_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensor>C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*

index_type0*
_output_shapes

:P
∆
&C_train/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
_output_shapes

:P*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P*
dtype0
Ч
-C_train/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign&C_train/Critic/eval_net/l1/w1_s/Adam_18C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
_output_shapes

:P*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(
і
+C_train/Critic/eval_net/l1/w1_s/Adam_1/readIdentity&C_train/Critic/eval_net/l1/w1_s/Adam_1*
_output_shapes

:P*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
Ј
6C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst**
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
ƒ
$C_train/Critic/eval_net/l1/w1_a/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a*
	container *
shape
:
С
+C_train/Critic/eval_net/l1/w1_a/Adam/AssignAssign$C_train/Critic/eval_net/l1/w1_a/Adam6C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
∞
)C_train/Critic/eval_net/l1/w1_a/Adam/readIdentity$C_train/Critic/eval_net/l1/w1_a/Adam*
_output_shapes

:*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
є
8C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst**
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
∆
&C_train/Critic/eval_net/l1/w1_a/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a*
	container *
shape
:
Ч
-C_train/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign&C_train/Critic/eval_net/l1/w1_a/Adam_18C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
і
+C_train/Critic/eval_net/l1/w1_a/Adam_1/readIdentity&C_train/Critic/eval_net/l1/w1_a/Adam_1*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
≥
4C_train/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*(
_class
loc:@Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
ј
"C_train/Critic/eval_net/l1/b1/Adam
VariableV2*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
Й
)C_train/Critic/eval_net/l1/b1/Adam/AssignAssign"C_train/Critic/eval_net/l1/b1/Adam4C_train/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
™
'C_train/Critic/eval_net/l1/b1/Adam/readIdentity"C_train/Critic/eval_net/l1/b1/Adam*
_output_shapes

:*
T0*(
_class
loc:@Critic/eval_net/l1/b1
µ
6C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*(
_class
loc:@Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
¬
$C_train/Critic/eval_net/l1/b1/Adam_1
VariableV2*(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
П
+C_train/Critic/eval_net/l1/b1/Adam_1/AssignAssign$C_train/Critic/eval_net/l1/b1/Adam_16C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
Ѓ
)C_train/Critic/eval_net/l1/b1/Adam_1/readIdentity$C_train/Critic/eval_net/l1/b1/Adam_1*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes

:
≈
=C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
“
+C_train/Critic/eval_net/q/dense/kernel/Adam
VariableV2*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
≠
2C_train/Critic/eval_net/q/dense/kernel/Adam/AssignAssign+C_train/Critic/eval_net/q/dense/kernel/Adam=C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
≈
0C_train/Critic/eval_net/q/dense/kernel/Adam/readIdentity+C_train/Critic/eval_net/q/dense/kernel/Adam*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
«
?C_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0
‘
-C_train/Critic/eval_net/q/dense/kernel/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
≥
4C_train/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign-C_train/Critic/eval_net/q/dense/kernel/Adam_1?C_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
…
2C_train/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity-C_train/Critic/eval_net/q/dense/kernel/Adam_1*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
є
;C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
∆
)C_train/Critic/eval_net/q/dense/bias/Adam
VariableV2*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
°
0C_train/Critic/eval_net/q/dense/bias/Adam/AssignAssign)C_train/Critic/eval_net/q/dense/bias/Adam;C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(
ї
.C_train/Critic/eval_net/q/dense/bias/Adam/readIdentity)C_train/Critic/eval_net/q/dense/bias/Adam*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
ї
=C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
»
+C_train/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:
І
2C_train/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign+C_train/Critic/eval_net/q/dense/bias/Adam_1=C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
њ
0C_train/Critic/eval_net/q/dense/bias/Adam_1/readIdentity+C_train/Critic/eval_net/q/dense/bias/Adam_1*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
_
C_train/Adam/learning_rateConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
W
C_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
W
C_train/Adam/beta2Const*
_output_shapes
: *
valueB
 *wЊ?*
dtype0
Y
C_train/Adam/epsilonConst*
_output_shapes
: *
valueB
 *wћ+2*
dtype0
А
5C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdamCritic/eval_net/l1/w1_s$C_train/Critic/eval_net/l1/w1_s/Adam&C_train/Critic/eval_net/l1/w1_s/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonKC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:P
В
5C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdamCritic/eval_net/l1/w1_a$C_train/Critic/eval_net/l1/w1_a/Adam&C_train/Critic/eval_net/l1/w1_a/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonMC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
х
3C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam	ApplyAdamCritic/eval_net/l1/b1"C_train/Critic/eval_net/l1/b1/Adam$C_train/Critic/eval_net/l1/b1/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonJC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*(
_class
loc:@Critic/eval_net/l1/b1*
use_nesterov( 
®
<C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdamCritic/eval_net/q/dense/kernel+C_train/Critic/eval_net/q/dense/kernel/Adam-C_train/Critic/eval_net/q/dense/kernel/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonPC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( 
Ы
:C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdamCritic/eval_net/q/dense/bias)C_train/Critic/eval_net/q/dense/bias/Adam+C_train/Critic/eval_net/q/dense/bias/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonQC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
≤
C_train/Adam/mulMulC_train/beta1_power/readC_train/Adam/beta14^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
Є
C_train/Adam/AssignAssignC_train/beta1_powerC_train/Adam/mul*
use_locking( *
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
і
C_train/Adam/mul_1MulC_train/beta2_power/readC_train/Adam/beta24^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam*
_output_shapes
: *
T0*(
_class
loc:@Critic/eval_net/l1/b1
Љ
C_train/Adam/Assign_1AssignC_train/beta2_powerC_train/Adam/mul_1*
use_locking( *
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
д
C_train/AdamNoOp^C_train/Adam/Assign^C_train/Adam/Assign_14^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam
u
a_grad/gradients/ShapeShapeCritic/eval_net/q/dense/BiasAdd*
_output_shapes
:*
T0*
out_type0
_
a_grad/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Х
a_grad/gradients/FillFilla_grad/gradients/Shapea_grad/gradients/grad_ys_0*'
_output_shapes
:€€€€€€€€€*
T0*

index_type0
£
Aa_grad/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrada_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
ў
;a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMula_grad/gradients/Fill#Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
∆
=a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMulCritic/eval_net/l1/Relua_grad/gradients/Fill*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
 
6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad;a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulCritic/eval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€
К
4a_grad/gradients/Critic/eval_net/l1/add_1_grad/ShapeShapeCritic/eval_net/l1/add*
_output_shapes
:*
T0*
out_type0
З
6a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
И
Da_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape6a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ч
2a_grad/gradients/Critic/eval_net/l1/add_1_grad/SumSum6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradDa_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
л
6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeReshape2a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ы
4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum_1Sum6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradFa_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
и
8a_grad/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum_16a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Л
2a_grad/gradients/Critic/eval_net/l1/add_grad/ShapeShapeCritic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
П
4a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1ShapeCritic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
В
Ba_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs2a_grad/gradients/Critic/eval_net/l1/add_grad/Shape4a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
у
0a_grad/gradients/Critic/eval_net/l1/add_grad/SumSum6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeBa_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
е
4a_grad/gradients/Critic/eval_net/l1/add_grad/ReshapeReshape0a_grad/gradients/Critic/eval_net/l1/add_grad/Sum2a_grad/gradients/Critic/eval_net/l1/add_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
ч
2a_grad/gradients/Critic/eval_net/l1/add_grad/Sum_1Sum6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeDa_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
л
6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1Reshape2a_grad/gradients/Critic/eval_net/l1/add_grad/Sum_14a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
р
8a_grad/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
а
:a_grad/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMulCritic/StopGradient6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
L
mul_8/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
^
mul_8Mulmul_8/xCritic/target_net/l1/w1_s/read*
_output_shapes

:P*
T0
L
mul_9/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
\
mul_9Mulmul_9/xCritic/eval_net/l1/w1_s/read*
T0*
_output_shapes

:P
C
add_4Addmul_8mul_9*
T0*
_output_shapes

:P
і
Assign_4AssignCritic/target_net/l1/w1_sadd_4*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
M
mul_10/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
`
mul_10Mulmul_10/xCritic/target_net/l1/w1_a/read*
_output_shapes

:*
T0
M
mul_11/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
^
mul_11Mulmul_11/xCritic/eval_net/l1/w1_a/read*
_output_shapes

:*
T0
E
add_5Addmul_10mul_11*
T0*
_output_shapes

:
і
Assign_5AssignCritic/target_net/l1/w1_aadd_5*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
M
mul_12/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
^
mul_12Mulmul_12/xCritic/target_net/l1/b1/read*
T0*
_output_shapes

:
M
mul_13/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
\
mul_13Mulmul_13/xCritic/eval_net/l1/b1/read*
_output_shapes

:*
T0
E
add_6Addmul_12mul_13*
T0*
_output_shapes

:
∞
Assign_6AssignCritic/target_net/l1/b1add_6**
_class 
loc:@Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
M
mul_14/xConst*
_output_shapes
: *
valueB
 *§p}?*
dtype0
g
mul_14Mulmul_14/x%Critic/target_net/q/dense/kernel/read*
_output_shapes

:*
T0
M
mul_15/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
e
mul_15Mulmul_15/x#Critic/eval_net/q/dense/kernel/read*
T0*
_output_shapes

:
E
add_7Addmul_14mul_15*
T0*
_output_shapes

:
¬
Assign_7Assign Critic/target_net/q/dense/kerneladd_7*
use_locking(*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
M
mul_16/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
a
mul_16Mulmul_16/x#Critic/target_net/q/dense/bias/read*
T0*
_output_shapes
:
M
mul_17/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
_
mul_17Mulmul_17/x!Critic/eval_net/q/dense/bias/read*
_output_shapes
:*
T0
A
add_8Addmul_16mul_17*
T0*
_output_shapes
:
Ї
Assign_8AssignCritic/target_net/q/dense/biasadd_8*
use_locking(*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
u
policy_grads/gradients/ShapeShapeActor/eval_net/a/scaled_a*
T0*
out_type0*
_output_shapes
:
e
 policy_grads/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
І
policy_grads/gradients/FillFillpolicy_grads/gradients/Shape policy_grads/gradients/grad_ys_0*'
_output_shapes
:€€€€€€€€€*
T0*

index_type0
Т
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/ShapeShapeActor/eval_net/a/a/Tanh*
out_type0*
_output_shapes
:*
T0
А
=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Э
Kpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ђ
9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/MulMulpolicy_grads/gradients/FillActor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
И
9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/SumSum9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/MulKpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
А
=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/ReshapeReshape9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
™
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Mul_1MulActor/eval_net/a/a/Tanhpolicy_grads/gradients/Fill*
T0*'
_output_shapes
:€€€€€€€€€
О
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum_1Sum;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Mul_1Mpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
х
?policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum_1=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
“
<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGradTanhGradActor/eval_net/a/a/Tanh=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
Ћ
Bpolicy_grads/gradients/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGrad<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGrad*
_output_shapes
:*
T0*
data_formatNHWC
ь
<policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMulMatMul<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGradActor/eval_net/a/a/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(*
T0
н
>policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMulActor/eval_net/l1/Relu<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGrad*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
ѕ
;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGradReluGrad<policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMulActor/eval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€
…
Apolicy_grads/gradients/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0
щ
;policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMulMatMul;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGradActor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€P*
transpose_a( *
transpose_b(
Ў
=policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGrad*
transpose_b( *
T0*
_output_shapes

:P*
transpose_a(
Т
!A_train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
£
A_train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias
“
A_train/beta1_power/AssignAssignA_train/beta1_power!A_train/beta1_power/initial_value**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ж
A_train/beta1_power/readIdentityA_train/beta1_power*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
Т
!A_train/beta2_power/initial_valueConst*
valueB
 *wЊ?**
_class 
loc:@Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
£
A_train/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container 
“
A_train/beta2_power/AssignAssignA_train/beta2_power!A_train/beta2_power/initial_value*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
Ж
A_train/beta2_power/readIdentityA_train/beta2_power*
_output_shapes
: *
T0**
_class 
loc:@Actor/eval_net/a/a/bias
≈
GA_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB"P      *
dtype0*
_output_shapes
:
ѓ
=A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ѓ
7A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillGA_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensor=A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*

index_type0*
_output_shapes

:P
∆
%A_train/Actor/eval_net/l1/kernel/Adam
VariableV2*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P*
dtype0*
_output_shapes

:P
Х
,A_train/Actor/eval_net/l1/kernel/Adam/AssignAssign%A_train/Actor/eval_net/l1/kernel/Adam7A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
_output_shapes

:P*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(
≥
*A_train/Actor/eval_net/l1/kernel/Adam/readIdentity%A_train/Actor/eval_net/l1/kernel/Adam*
_output_shapes

:P*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
«
IA_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB"P      *
dtype0*
_output_shapes
:
±
?A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
µ
9A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillIA_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensor?A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*

index_type0*
_output_shapes

:P
»
'A_train/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P
Ы
.A_train/Actor/eval_net/l1/kernel/Adam_1/AssignAssign'A_train/Actor/eval_net/l1/kernel/Adam_19A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:P
Ј
,A_train/Actor/eval_net/l1/kernel/Adam_1/readIdentity'A_train/Actor/eval_net/l1/kernel/Adam_1*
_output_shapes

:P*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
≠
5A_train/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*)
_class
loc:@Actor/eval_net/l1/bias*
valueB*    *
dtype0*
_output_shapes
:
Ї
#A_train/Actor/eval_net/l1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:
Й
*A_train/Actor/eval_net/l1/bias/Adam/AssignAssign#A_train/Actor/eval_net/l1/bias/Adam5A_train/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
©
(A_train/Actor/eval_net/l1/bias/Adam/readIdentity#A_train/Actor/eval_net/l1/bias/Adam*)
_class
loc:@Actor/eval_net/l1/bias*
_output_shapes
:*
T0
ѓ
7A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*)
_class
loc:@Actor/eval_net/l1/bias*
valueB*    *
dtype0*
_output_shapes
:
Љ
%A_train/Actor/eval_net/l1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:
П
,A_train/Actor/eval_net/l1/bias/Adam_1/AssignAssign%A_train/Actor/eval_net/l1/bias/Adam_17A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
≠
*A_train/Actor/eval_net/l1/bias/Adam_1/readIdentity%A_train/Actor/eval_net/l1/bias/Adam_1*
_output_shapes
:*
T0*)
_class
loc:@Actor/eval_net/l1/bias
ї
8A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB*    *
dtype0*
_output_shapes

:
»
&A_train/Actor/eval_net/a/a/kernel/Adam
VariableV2*
shared_name *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
Щ
-A_train/Actor/eval_net/a/a/kernel/Adam/AssignAssign&A_train/Actor/eval_net/a/a/kernel/Adam8A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
ґ
+A_train/Actor/eval_net/a/a/kernel/Adam/readIdentity&A_train/Actor/eval_net/a/a/kernel/Adam*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
љ
:A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB*    *
dtype0*
_output_shapes

:
 
(A_train/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
Я
/A_train/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign(A_train/Actor/eval_net/a/a/kernel/Adam_1:A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Ї
-A_train/Actor/eval_net/a/a/kernel/Adam_1/readIdentity(A_train/Actor/eval_net/a/a/kernel/Adam_1*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
ѓ
6A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst**
_class 
loc:@Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
Љ
$A_train/Actor/eval_net/a/a/bias/Adam
VariableV2*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Н
+A_train/Actor/eval_net/a/a/bias/Adam/AssignAssign$A_train/Actor/eval_net/a/a/bias/Adam6A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
ђ
)A_train/Actor/eval_net/a/a/bias/Adam/readIdentity$A_train/Actor/eval_net/a/a/bias/Adam*
_output_shapes
:*
T0**
_class 
loc:@Actor/eval_net/a/a/bias
±
8A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:**
_class 
loc:@Actor/eval_net/a/a/bias*
valueB*    
Њ
&A_train/Actor/eval_net/a/a/bias/Adam_1
VariableV2*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
У
-A_train/Actor/eval_net/a/a/bias/Adam_1/AssignAssign&A_train/Actor/eval_net/a/a/bias/Adam_18A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
∞
+A_train/Actor/eval_net/a/a/bias/Adam_1/readIdentity&A_train/Actor/eval_net/a/a/bias/Adam_1*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
:
_
A_train/Adam/learning_rateConst*
valueB
 *oГЇ*
dtype0*
_output_shapes
: 
W
A_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
W
A_train/Adam/beta2Const*
_output_shapes
: *
valueB
 *wЊ?*
dtype0
Y
A_train/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
ч
6A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdamActor/eval_net/l1/kernel%A_train/Actor/eval_net/l1/kernel/Adam'A_train/Actor/eval_net/l1/kernel/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilon=policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_locking( *
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes

:P
н
4A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam	ApplyAdamActor/eval_net/l1/bias#A_train/Actor/eval_net/l1/bias/Adam%A_train/Actor/eval_net/l1/bias/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilonApolicy_grads/gradients/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*)
_class
loc:@Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
э
7A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdamActor/eval_net/a/a/kernel&A_train/Actor/eval_net/a/a/kernel/Adam(A_train/Actor/eval_net/a/a/kernel/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilon>policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
у
5A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdamActor/eval_net/a/a/bias$A_train/Actor/eval_net/a/a/bias/Adam&A_train/Actor/eval_net/a/a/bias/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilonBpolicy_grads/gradients/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:
ф
A_train/Adam/mulMulA_train/beta1_power/readA_train/Adam/beta16^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
Ї
A_train/Adam/AssignAssignA_train/beta1_powerA_train/Adam/mul*
use_locking( *
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
ц
A_train/Adam/mul_1MulA_train/beta2_power/readA_train/Adam/beta26^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam*
_output_shapes
: *
T0**
_class 
loc:@Actor/eval_net/a/a/bias
Њ
A_train/Adam/Assign_1AssignA_train/beta2_powerA_train/Adam/mul_1**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
§
A_train/AdamNoOp^A_train/Adam/Assign^A_train/Adam/Assign_16^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam
в
initNoOp,^A_train/Actor/eval_net/a/a/bias/Adam/Assign.^A_train/Actor/eval_net/a/a/bias/Adam_1/Assign.^A_train/Actor/eval_net/a/a/kernel/Adam/Assign0^A_train/Actor/eval_net/a/a/kernel/Adam_1/Assign+^A_train/Actor/eval_net/l1/bias/Adam/Assign-^A_train/Actor/eval_net/l1/bias/Adam_1/Assign-^A_train/Actor/eval_net/l1/kernel/Adam/Assign/^A_train/Actor/eval_net/l1/kernel/Adam_1/Assign^A_train/beta1_power/Assign^A_train/beta2_power/Assign^Actor/eval_net/a/a/bias/Assign!^Actor/eval_net/a/a/kernel/Assign^Actor/eval_net/l1/bias/Assign ^Actor/eval_net/l1/kernel/Assign!^Actor/target_net/a/a/bias/Assign#^Actor/target_net/a/a/kernel/Assign ^Actor/target_net/l1/bias/Assign"^Actor/target_net/l1/kernel/Assign*^C_train/Critic/eval_net/l1/b1/Adam/Assign,^C_train/Critic/eval_net/l1/b1/Adam_1/Assign,^C_train/Critic/eval_net/l1/w1_a/Adam/Assign.^C_train/Critic/eval_net/l1/w1_a/Adam_1/Assign,^C_train/Critic/eval_net/l1/w1_s/Adam/Assign.^C_train/Critic/eval_net/l1/w1_s/Adam_1/Assign1^C_train/Critic/eval_net/q/dense/bias/Adam/Assign3^C_train/Critic/eval_net/q/dense/bias/Adam_1/Assign3^C_train/Critic/eval_net/q/dense/kernel/Adam/Assign5^C_train/Critic/eval_net/q/dense/kernel/Adam_1/Assign^C_train/beta1_power/Assign^C_train/beta2_power/Assign^Critic/eval_net/l1/b1/Assign^Critic/eval_net/l1/w1_a/Assign^Critic/eval_net/l1/w1_s/Assign$^Critic/eval_net/q/dense/bias/Assign&^Critic/eval_net/q/dense/kernel/Assign^Critic/target_net/l1/b1/Assign!^Critic/target_net/l1/w1_a/Assign!^Critic/target_net/l1/w1_s/Assign&^Critic/target_net/q/dense/bias/Assign(^Critic/target_net/q/dense/kernel/Assign"&bЗньЫХ     e»ъS	С√Ћt„AJОЂ
©В
:
Add
x"T
y"T
z"T"
Ttype:
2	
о
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	Р
2
StopGradient

input"T
output"T"	
Ttype
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*1.14.02v1.14.0-rc1-22-gaf24dc91b5х“
f
S/sPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€P*
shape:€€€€€€€€€P
f
R/rPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
S_/s_Placeholder*
shape:€€€€€€€€€P*
dtype0*'
_output_shapes
:€€€€€€€€€P
ґ
8Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB"P      *
dtype0*
_output_shapes
:
©
7Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ђ
9Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB
 *ЪЩЩ>*
dtype0
Х
GActor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal8Actor/eval_net/l1/kernel/Initializer/random_normal/shape*

seed*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
seed2*
dtype0*
_output_shapes

:P
Ч
6Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulGActor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal9Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
_output_shapes

:P*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
А
2Actor/eval_net/l1/kernel/Initializer/random_normalAdd6Actor/eval_net/l1/kernel/Initializer/random_normal/mul7Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
є
Actor/eval_net/l1/kernel
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P
ц
Actor/eval_net/l1/kernel/AssignAssignActor/eval_net/l1/kernel2Actor/eval_net/l1/kernel/Initializer/random_normal*
_output_shapes

:P*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(
Щ
Actor/eval_net/l1/kernel/readIdentityActor/eval_net/l1/kernel*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
†
(Actor/eval_net/l1/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*)
_class
loc:@Actor/eval_net/l1/bias*
valueB*Ќћћ=
≠
Actor/eval_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:
в
Actor/eval_net/l1/bias/AssignAssignActor/eval_net/l1/bias(Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
П
Actor/eval_net/l1/bias/readIdentityActor/eval_net/l1/bias*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
_output_shapes
:
Ю
Actor/eval_net/l1/MatMulMatMulS/sActor/eval_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( *
T0
§
Actor/eval_net/l1/BiasAddBiasAddActor/eval_net/l1/MatMulActor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
k
Actor/eval_net/l1/ReluReluActor/eval_net/l1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
Є
9Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ђ
8Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
≠
:Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB
 *ЪЩЩ>*
dtype0
Ш
HActor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
seed2
Ы
7Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulHActor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal:Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
Д
3Actor/eval_net/a/a/kernel/Initializer/random_normalAdd7Actor/eval_net/a/a/kernel/Initializer/random_normal/mul8Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
ї
Actor/eval_net/a/a/kernel
VariableV2*
shared_name *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
ъ
 Actor/eval_net/a/a/kernel/AssignAssignActor/eval_net/a/a/kernel3Actor/eval_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Ь
Actor/eval_net/a/a/kernel/readIdentityActor/eval_net/a/a/kernel*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
Ґ
)Actor/eval_net/a/a/bias/Initializer/ConstConst**
_class 
loc:@Actor/eval_net/a/a/bias*
valueB*Ќћћ=*
dtype0*
_output_shapes
:
ѓ
Actor/eval_net/a/a/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:
ж
Actor/eval_net/a/a/bias/AssignAssignActor/eval_net/a/a/bias)Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Т
Actor/eval_net/a/a/bias/readIdentityActor/eval_net/a/a/bias*
_output_shapes
:*
T0**
_class 
loc:@Actor/eval_net/a/a/bias
≥
Actor/eval_net/a/a/MatMulMatMulActor/eval_net/l1/ReluActor/eval_net/a/a/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( *
T0
І
Actor/eval_net/a/a/BiasAddBiasAddActor/eval_net/a/a/MatMulActor/eval_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
m
Actor/eval_net/a/a/TanhTanhActor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
`
Actor/eval_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
И
Actor/eval_net/a/scaled_aMulActor/eval_net/a/a/TanhActor/eval_net/a/scaled_a/y*'
_output_shapes
:€€€€€€€€€*
T0
Ї
:Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*-
_class#
!loc:@Actor/target_net/l1/kernel*
valueB"P      *
dtype0
≠
9Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*-
_class#
!loc:@Actor/target_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ѓ
;Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@Actor/target_net/l1/kernel*
valueB
 *ЪЩЩ>
Ы
IActor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:Actor/target_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:P*

seed*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
seed2(
Я
8Actor/target_net/l1/kernel/Initializer/random_normal/mulMulIActor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
_output_shapes

:P
И
4Actor/target_net/l1/kernel/Initializer/random_normalAdd8Actor/target_net/l1/kernel/Initializer/random_normal/mul9Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
_output_shapes

:P
љ
Actor/target_net/l1/kernel
VariableV2*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name *-
_class#
!loc:@Actor/target_net/l1/kernel
ю
!Actor/target_net/l1/kernel/AssignAssignActor/target_net/l1/kernel4Actor/target_net/l1/kernel/Initializer/random_normal*-
_class#
!loc:@Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0
Я
Actor/target_net/l1/kernel/readIdentityActor/target_net/l1/kernel*-
_class#
!loc:@Actor/target_net/l1/kernel*
_output_shapes

:P*
T0
§
*Actor/target_net/l1/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*+
_class!
loc:@Actor/target_net/l1/bias*
valueB*Ќћћ=
±
Actor/target_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@Actor/target_net/l1/bias*
	container *
shape:
к
Actor/target_net/l1/bias/AssignAssignActor/target_net/l1/bias*Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
Х
Actor/target_net/l1/bias/readIdentityActor/target_net/l1/bias*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
_output_shapes
:
§
Actor/target_net/l1/MatMulMatMulS_/s_Actor/target_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
™
Actor/target_net/l1/BiasAddBiasAddActor/target_net/l1/MatMulActor/target_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
o
Actor/target_net/l1/ReluReluActor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Љ
;Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*.
_class$
" loc:@Actor/target_net/a/a/kernel*
valueB"      *
dtype0*
_output_shapes
:
ѓ
:Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*.
_class$
" loc:@Actor/target_net/a/a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
±
<Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@Actor/target_net/a/a/kernel*
valueB
 *ЪЩЩ>*
dtype0
Ю
JActor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
seed28*
dtype0
£
9Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulJActor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:*
T0
М
5Actor/target_net/a/a/kernel/Initializer/random_normalAdd9Actor/target_net/a/a/kernel/Initializer/random_normal/mul:Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:
њ
Actor/target_net/a/a/kernel
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@Actor/target_net/a/a/kernel
В
"Actor/target_net/a/a/kernel/AssignAssignActor/target_net/a/a/kernel5Actor/target_net/a/a/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
validate_shape(
Ґ
 Actor/target_net/a/a/kernel/readIdentityActor/target_net/a/a/kernel*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:*
T0
¶
+Actor/target_net/a/a/bias/Initializer/ConstConst*,
_class"
 loc:@Actor/target_net/a/a/bias*
valueB*Ќћћ=*
dtype0*
_output_shapes
:
≥
Actor/target_net/a/a/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@Actor/target_net/a/a/bias*
	container *
shape:
о
 Actor/target_net/a/a/bias/AssignAssignActor/target_net/a/a/bias+Actor/target_net/a/a/bias/Initializer/Const*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ш
Actor/target_net/a/a/bias/readIdentityActor/target_net/a/a/bias*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias*
_output_shapes
:
є
Actor/target_net/a/a/MatMulMatMulActor/target_net/l1/Relu Actor/target_net/a/a/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( *
T0
≠
Actor/target_net/a/a/BiasAddBiasAddActor/target_net/a/a/MatMulActor/target_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
q
Actor/target_net/a/a/TanhTanhActor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
b
Actor/target_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
О
Actor/target_net/a/scaled_aMulActor/target_net/a/a/TanhActor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
J
mul/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
[
mulMulmul/xActor/target_net/l1/kernel/read*
_output_shapes

:P*
T0
L
mul_1/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
]
mul_1Mulmul_1/xActor/eval_net/l1/kernel/read*
T0*
_output_shapes

:P
?
addAddmulmul_1*
T0*
_output_shapes

:P
≤
AssignAssignActor/target_net/l1/kerneladd*
use_locking(*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:P
L
mul_2/xConst*
_output_shapes
: *
valueB
 *§p}?*
dtype0
Y
mul_2Mulmul_2/xActor/target_net/l1/bias/read*
T0*
_output_shapes
:
L
mul_3/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
W
mul_3Mulmul_3/xActor/eval_net/l1/bias/read*
_output_shapes
:*
T0
?
add_1Addmul_2mul_3*
_output_shapes
:*
T0
Ѓ
Assign_1AssignActor/target_net/l1/biasadd_1*
use_locking(*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
L
mul_4/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
`
mul_4Mulmul_4/x Actor/target_net/a/a/kernel/read*
T0*
_output_shapes

:
L
mul_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
^
mul_5Mulmul_5/xActor/eval_net/a/a/kernel/read*
T0*
_output_shapes

:
C
add_2Addmul_4mul_5*
_output_shapes

:*
T0
Є
Assign_2AssignActor/target_net/a/a/kerneladd_2*.
_class$
" loc:@Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
L
mul_6/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
Z
mul_6Mulmul_6/xActor/target_net/a/a/bias/read*
T0*
_output_shapes
:
L
mul_7/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
X
mul_7Mulmul_7/xActor/eval_net/a/a/bias/read*
T0*
_output_shapes
:
?
add_3Addmul_6mul_7*
_output_shapes
:*
T0
∞
Assign_3AssignActor/target_net/a/a/biasadd_3*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
p
Critic/StopGradientStopGradientActor/eval_net/a/scaled_a*
T0*'
_output_shapes
:€€€€€€€€€
і
7Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB"P      *
dtype0*
_output_shapes
:
І
6Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: **
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB
 *    
©
8Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
Т
FCritic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
seed2c*
dtype0*
_output_shapes

:P*

seed
У
5Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulFCritic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal8Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
_output_shapes

:P*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
ь
1Critic/eval_net/l1/w1_s/Initializer/random_normalAdd5Critic/eval_net/l1/w1_s/Initializer/random_normal/mul6Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
Ј
Critic/eval_net/l1/w1_s
VariableV2*
_output_shapes

:P*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P*
dtype0
т
Critic/eval_net/l1/w1_s/AssignAssignCritic/eval_net/l1/w1_s1Critic/eval_net/l1/w1_s/Initializer/random_normal*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:P*
use_locking(
Ц
Critic/eval_net/l1/w1_s/readIdentityCritic/eval_net/l1/w1_s*
_output_shapes

:P*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
і
7Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:**
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB"      
І
6Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: **
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB
 *    
©
8Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst**
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
Т
FCritic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*

seed*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
seed2l*
dtype0*
_output_shapes

:
У
5Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulFCritic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal8Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
ь
1Critic/eval_net/l1/w1_a/Initializer/random_normalAdd5Critic/eval_net/l1/w1_a/Initializer/random_normal/mul6Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
_output_shapes

:*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
Ј
Critic/eval_net/l1/w1_a
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a*
	container *
shape
:
т
Critic/eval_net/l1/w1_a/AssignAssignCritic/eval_net/l1/w1_a1Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ц
Critic/eval_net/l1/w1_a/readIdentityCritic/eval_net/l1/w1_a*
_output_shapes

:*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
¶
'Critic/eval_net/l1/b1/Initializer/ConstConst*(
_class
loc:@Critic/eval_net/l1/b1*
valueB*Ќћћ=*
dtype0*
_output_shapes

:
≥
Critic/eval_net/l1/b1
VariableV2*(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
в
Critic/eval_net/l1/b1/AssignAssignCritic/eval_net/l1/b1'Critic/eval_net/l1/b1/Initializer/Const*
_output_shapes

:*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(
Р
Critic/eval_net/l1/b1/readIdentityCritic/eval_net/l1/b1*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes

:
Ю
Critic/eval_net/l1/MatMulMatMulS/sCritic/eval_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
∞
Critic/eval_net/l1/MatMul_1MatMulCritic/StopGradientCritic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
З
Critic/eval_net/l1/addAddCritic/eval_net/l1/MatMulCritic/eval_net/l1/MatMul_1*'
_output_shapes
:€€€€€€€€€*
T0
Е
Critic/eval_net/l1/add_1AddCritic/eval_net/l1/addCritic/eval_net/l1/b1/read*
T0*'
_output_shapes
:€€€€€€€€€
k
Critic/eval_net/l1/ReluReluCritic/eval_net/l1/add_1*
T0*'
_output_shapes
:€€€€€€€€€
¬
>Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
µ
=Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB
 *    *
dtype0
Ј
?Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
І
MCritic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal>Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
seed2~*
dtype0
ѓ
<Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulMCritic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormal?Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
Ш
8Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd<Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul=Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
≈
Critic/eval_net/q/dense/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container *
shape
:
О
%Critic/eval_net/q/dense/kernel/AssignAssignCritic/eval_net/q/dense/kernel8Critic/eval_net/q/dense/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
Ђ
#Critic/eval_net/q/dense/kernel/readIdentityCritic/eval_net/q/dense/kernel*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
ђ
.Critic/eval_net/q/dense/bias/Initializer/ConstConst*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
valueB*Ќћћ=*
dtype0*
_output_shapes
:
є
Critic/eval_net/q/dense/bias
VariableV2*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ъ
#Critic/eval_net/q/dense/bias/AssignAssignCritic/eval_net/q/dense/bias.Critic/eval_net/q/dense/bias/Initializer/Const*
_output_shapes
:*
use_locking(*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(
°
!Critic/eval_net/q/dense/bias/readIdentityCritic/eval_net/q/dense/bias*
_output_shapes
:*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias
Њ
Critic/eval_net/q/dense/MatMulMatMulCritic/eval_net/l1/Relu#Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
ґ
Critic/eval_net/q/dense/BiasAddBiasAddCritic/eval_net/q/dense/MatMul!Critic/eval_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
Є
9Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*,
_class"
 loc:@Critic/target_net/l1/w1_s*
valueB"P      *
dtype0*
_output_shapes
:
Ђ
8Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*,
_class"
 loc:@Critic/target_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
≠
:Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*,
_class"
 loc:@Critic/target_net/l1/w1_s*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
Щ
HCritic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
seed2Н*
dtype0*
_output_shapes

:P*

seed
Ы
7Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulHCritic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P*
T0
Д
3Critic/target_net/l1/w1_s/Initializer/random_normalAdd7Critic/target_net/l1/w1_s/Initializer/random_normal/mul8Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P
ї
Critic/target_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *,
_class"
 loc:@Critic/target_net/l1/w1_s*
	container *
shape
:P
ъ
 Critic/target_net/l1/w1_s/AssignAssignCritic/target_net/l1/w1_s3Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
Ь
Critic/target_net/l1/w1_s/readIdentityCritic/target_net/l1/w1_s*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P
Є
9Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*,
_class"
 loc:@Critic/target_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
Ђ
8Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
_output_shapes
: *,
_class"
 loc:@Critic/target_net/l1/w1_a*
valueB
 *    *
dtype0
≠
:Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@Critic/target_net/l1/w1_a*
valueB
 *Ќћћ=
Щ
HCritic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
seed2Ц
Ы
7Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulHCritic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a
Д
3Critic/target_net/l1/w1_a/Initializer/random_normalAdd7Critic/target_net/l1/w1_a/Initializer/random_normal/mul8Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
_output_shapes

:
ї
Critic/target_net/l1/w1_a
VariableV2*
shared_name *,
_class"
 loc:@Critic/target_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:
ъ
 Critic/target_net/l1/w1_a/AssignAssignCritic/target_net/l1/w1_a3Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ь
Critic/target_net/l1/w1_a/readIdentityCritic/target_net/l1/w1_a*
_output_shapes

:*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a
™
)Critic/target_net/l1/b1/Initializer/ConstConst**
_class 
loc:@Critic/target_net/l1/b1*
valueB*Ќћћ=*
dtype0*
_output_shapes

:
Ј
Critic/target_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/target_net/l1/b1*
	container *
shape
:
к
Critic/target_net/l1/b1/AssignAssignCritic/target_net/l1/b1)Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
Ц
Critic/target_net/l1/b1/readIdentityCritic/target_net/l1/b1*
T0**
_class 
loc:@Critic/target_net/l1/b1*
_output_shapes

:
§
Critic/target_net/l1/MatMulMatMulS_/s_Critic/target_net/l1/w1_s/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( *
T0
Љ
Critic/target_net/l1/MatMul_1MatMulActor/target_net/a/scaled_aCritic/target_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
Н
Critic/target_net/l1/addAddCritic/target_net/l1/MatMulCritic/target_net/l1/MatMul_1*'
_output_shapes
:€€€€€€€€€*
T0
Л
Critic/target_net/l1/add_1AddCritic/target_net/l1/addCritic/target_net/l1/b1/read*
T0*'
_output_shapes
:€€€€€€€€€
o
Critic/target_net/l1/ReluReluCritic/target_net/l1/add_1*'
_output_shapes
:€€€€€€€€€*
T0
∆
@Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
valueB"      *
dtype0
є
?Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
valueB
 *    
ї
ACritic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
Ѓ
OCritic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
seed2®*
dtype0
Ј
>Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulOCritic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalACritic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
_output_shapes

:
†
:Critic/target_net/q/dense/kernel/Initializer/random_normalAdd>Critic/target_net/q/dense/kernel/Initializer/random_normal/mul?Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
_output_shapes

:
…
 Critic/target_net/q/dense/kernel
VariableV2*
_output_shapes

:*
shared_name *3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
	container *
shape
:*
dtype0
Ц
'Critic/target_net/q/dense/kernel/AssignAssign Critic/target_net/q/dense/kernel:Critic/target_net/q/dense/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel
±
%Critic/target_net/q/dense/kernel/readIdentity Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel
∞
0Critic/target_net/q/dense/bias/Initializer/ConstConst*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
valueB*Ќћћ=*
dtype0*
_output_shapes
:
љ
Critic/target_net/q/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@Critic/target_net/q/dense/bias*
	container *
shape:
В
%Critic/target_net/q/dense/bias/AssignAssignCritic/target_net/q/dense/bias0Critic/target_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
І
#Critic/target_net/q/dense/bias/readIdentityCritic/target_net/q/dense/bias*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
_output_shapes
:*
T0
ƒ
 Critic/target_net/q/dense/MatMulMatMulCritic/target_net/l1/Relu%Critic/target_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
Љ
!Critic/target_net/q/dense/BiasAddBiasAdd Critic/target_net/q/dense/MatMul#Critic/target_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
S
target_q/mul/xConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
x
target_q/mulMultarget_q/mul/x!Critic/target_net/q/dense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
X
target_q/addAddR/rtarget_q/mul*'
_output_shapes
:€€€€€€€€€*
T0
Р
TD_error/SquaredDifferenceSquaredDifferencetarget_q/addCritic/eval_net/q/dense/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
_
TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

TD_error/MeanMeanTD_error/SquaredDifferenceTD_error/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Z
C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
`
C_train/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
З
C_train/gradients/FillFillC_train/gradients/ShapeC_train/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
Г
2C_train/gradients/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ї
,C_train/gradients/TD_error/Mean_grad/ReshapeReshapeC_train/gradients/Fill2C_train/gradients/TD_error/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
Д
*C_train/gradients/TD_error/Mean_grad/ShapeShapeTD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
ѕ
)C_train/gradients/TD_error/Mean_grad/TileTile,C_train/gradients/TD_error/Mean_grad/Reshape*C_train/gradients/TD_error/Mean_grad/Shape*'
_output_shapes
:€€€€€€€€€*

Tmultiples0*
T0
Ж
,C_train/gradients/TD_error/Mean_grad/Shape_1ShapeTD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
o
,C_train/gradients/TD_error/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
t
*C_train/gradients/TD_error/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
…
)C_train/gradients/TD_error/Mean_grad/ProdProd,C_train/gradients/TD_error/Mean_grad/Shape_1*C_train/gradients/TD_error/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
v
,C_train/gradients/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ќ
+C_train/gradients/TD_error/Mean_grad/Prod_1Prod,C_train/gradients/TD_error/Mean_grad/Shape_2,C_train/gradients/TD_error/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
p
.C_train/gradients/TD_error/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
µ
,C_train/gradients/TD_error/Mean_grad/MaximumMaximum+C_train/gradients/TD_error/Mean_grad/Prod_1.C_train/gradients/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
≥
-C_train/gradients/TD_error/Mean_grad/floordivFloorDiv)C_train/gradients/TD_error/Mean_grad/Prod,C_train/gradients/TD_error/Mean_grad/Maximum*
_output_shapes
: *
T0
†
)C_train/gradients/TD_error/Mean_grad/CastCast-C_train/gradients/TD_error/Mean_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
њ
,C_train/gradients/TD_error/Mean_grad/truedivRealDiv)C_train/gradients/TD_error/Mean_grad/Tile)C_train/gradients/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
Г
7C_train/gradients/TD_error/SquaredDifference_grad/ShapeShapetarget_q/add*
_output_shapes
:*
T0*
out_type0
Ш
9C_train/gradients/TD_error/SquaredDifference_grad/Shape_1ShapeCritic/eval_net/q/dense/BiasAdd*
_output_shapes
:*
T0*
out_type0
С
GC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs7C_train/gradients/TD_error/SquaredDifference_grad/Shape9C_train/gradients/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ђ
8C_train/gradients/TD_error/SquaredDifference_grad/scalarConst-^C_train/gradients/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
÷
5C_train/gradients/TD_error/SquaredDifference_grad/MulMul8C_train/gradients/TD_error/SquaredDifference_grad/scalar,C_train/gradients/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
ћ
5C_train/gradients/TD_error/SquaredDifference_grad/subSubtarget_q/addCritic/eval_net/q/dense/BiasAdd-^C_train/gradients/TD_error/Mean_grad/truediv*'
_output_shapes
:€€€€€€€€€*
T0
ё
7C_train/gradients/TD_error/SquaredDifference_grad/mul_1Mul5C_train/gradients/TD_error/SquaredDifference_grad/Mul5C_train/gradients/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:€€€€€€€€€
ю
5C_train/gradients/TD_error/SquaredDifference_grad/SumSum7C_train/gradients/TD_error/SquaredDifference_grad/mul_1GC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ф
9C_train/gradients/TD_error/SquaredDifference_grad/ReshapeReshape5C_train/gradients/TD_error/SquaredDifference_grad/Sum7C_train/gradients/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
В
7C_train/gradients/TD_error/SquaredDifference_grad/Sum_1Sum7C_train/gradients/TD_error/SquaredDifference_grad/mul_1IC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ъ
;C_train/gradients/TD_error/SquaredDifference_grad/Reshape_1Reshape7C_train/gradients/TD_error/SquaredDifference_grad/Sum_19C_train/gradients/TD_error/SquaredDifference_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Ђ
5C_train/gradients/TD_error/SquaredDifference_grad/NegNeg;C_train/gradients/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
Њ
BC_train/gradients/TD_error/SquaredDifference_grad/tuple/group_depsNoOp6^C_train/gradients/TD_error/SquaredDifference_grad/Neg:^C_train/gradients/TD_error/SquaredDifference_grad/Reshape
÷
JC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity9C_train/gradients/TD_error/SquaredDifference_grad/ReshapeC^C_train/gradients/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:€€€€€€€€€
–
LC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity5C_train/gradients/TD_error/SquaredDifference_grad/NegC^C_train/gradients/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*H
_class>
<:loc:@C_train/gradients/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€
џ
BC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradLC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
г
GC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpC^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradM^C_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1
п
OC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityLC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1H^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*H
_class>
<:loc:@C_train/gradients/TD_error/SquaredDifference_grad/Neg
з
QC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityBC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradH^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ф
<C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulOC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency#Critic/eval_net/q/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
Б
>C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMulCritic/eval_net/l1/ReluOC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
ќ
FC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOp=^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul?^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
д
NC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity<C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulG^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€
б
PC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1Identity>C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1G^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
ё
7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradReluGradNC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyCritic/eval_net/l1/Relu*'
_output_shapes
:€€€€€€€€€*
T0
Л
5C_train/gradients/Critic/eval_net/l1/add_1_grad/ShapeShapeCritic/eval_net/l1/add*
out_type0*
_output_shapes
:*
T0
И
7C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB"      *
dtype0
Л
EC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs5C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape7C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ъ
3C_train/gradients/Critic/eval_net/l1/add_1_grad/SumSum7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradEC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
о
7C_train/gradients/Critic/eval_net/l1/add_1_grad/ReshapeReshape3C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum5C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
ю
5C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum_1Sum7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradGC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
л
9C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape5C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum_17C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Њ
@C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp8^C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape:^C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1
ќ
HC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity7C_train/gradients/Critic/eval_net/l1/add_1_grad/ReshapeA^C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Ћ
JC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity9C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1A^C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1
М
3C_train/gradients/Critic/eval_net/l1/add_grad/ShapeShapeCritic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
Р
5C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1ShapeCritic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
Е
CC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs3C_train/gradients/Critic/eval_net/l1/add_grad/Shape5C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
З
1C_train/gradients/Critic/eval_net/l1/add_grad/SumSumHC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyCC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
и
5C_train/gradients/Critic/eval_net/l1/add_grad/ReshapeReshape1C_train/gradients/Critic/eval_net/l1/add_grad/Sum3C_train/gradients/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Л
3C_train/gradients/Critic/eval_net/l1/add_grad/Sum_1SumHC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyEC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
о
7C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1Reshape3C_train/gradients/Critic/eval_net/l1/add_grad/Sum_15C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Є
>C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp6^C_train/gradients/Critic/eval_net/l1/add_grad/Reshape8^C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1
∆
FC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity5C_train/gradients/Critic/eval_net/l1/add_grad/Reshape?^C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@C_train/gradients/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€
ћ
HC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity7C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1?^C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
€
7C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMulMatMulFC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependencyCritic/eval_net/l1/w1_s/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€P
я
9C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sFC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:P
њ
AC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp8^C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul:^C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1
–
IC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity7C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMulB^C_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€P*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul
Ќ
KC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity9C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1B^C_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:P
Г
9C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulHC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(
у
;C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMulCritic/StopGradientHC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
≈
CC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp:^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul<^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
Ў
KC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity9C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulD^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:€€€€€€€€€
’
MC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity;C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1D^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:
Р
!C_train/beta1_power/initial_valueConst*(
_class
loc:@Critic/eval_net/l1/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
°
C_train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape: 
–
C_train/beta1_power/AssignAssignC_train/beta1_power!C_train/beta1_power/initial_value*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
Д
C_train/beta1_power/readIdentityC_train/beta1_power*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
Р
!C_train/beta2_power/initial_valueConst*(
_class
loc:@Critic/eval_net/l1/b1*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
°
C_train/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container 
–
C_train/beta2_power/AssignAssignC_train/beta2_power!C_train/beta2_power/initial_value*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Д
C_train/beta2_power/readIdentityC_train/beta2_power*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
√
FC_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"P      **
_class 
loc:@Critic/eval_net/l1/w1_s
≠
<C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst*
valueB
 *    **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
Ђ
6C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillFC_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor<C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*
T0*

index_type0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
ƒ
$C_train/Critic/eval_net/l1/w1_s/Adam
VariableV2*
dtype0*
_output_shapes

:P*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P
С
+C_train/Critic/eval_net/l1/w1_s/Adam/AssignAssign$C_train/Critic/eval_net/l1/w1_s/Adam6C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:P*
use_locking(
∞
)C_train/Critic/eval_net/l1/w1_s/Adam/readIdentity$C_train/Critic/eval_net/l1/w1_s/Adam*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
≈
HC_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"P      **
_class 
loc:@Critic/eval_net/l1/w1_s
ѓ
>C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
±
8C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillHC_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensor>C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
T0*

index_type0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
∆
&C_train/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_s
Ч
-C_train/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign&C_train/Critic/eval_net/l1/w1_s/Adam_18C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
і
+C_train/Critic/eval_net/l1/w1_s/Adam_1/readIdentity&C_train/Critic/eval_net/l1/w1_s/Adam_1*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
Ј
6C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
valueB*    **
_class 
loc:@Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
ƒ
$C_train/Critic/eval_net/l1/w1_a/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a
С
+C_train/Critic/eval_net/l1/w1_a/Adam/AssignAssign$C_train/Critic/eval_net/l1/w1_a/Adam6C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(
∞
)C_train/Critic/eval_net/l1/w1_a/Adam/readIdentity$C_train/Critic/eval_net/l1/w1_a/Adam*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
є
8C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    **
_class 
loc:@Critic/eval_net/l1/w1_a
∆
&C_train/Critic/eval_net/l1/w1_a/Adam_1
VariableV2**
_class 
loc:@Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ч
-C_train/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign&C_train/Critic/eval_net/l1/w1_a/Adam_18C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
і
+C_train/Critic/eval_net/l1/w1_a/Adam_1/readIdentity&C_train/Critic/eval_net/l1/w1_a/Adam_1*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
≥
4C_train/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
_output_shapes

:*
valueB*    *(
_class
loc:@Critic/eval_net/l1/b1*
dtype0
ј
"C_train/Critic/eval_net/l1/b1/Adam
VariableV2*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
Й
)C_train/Critic/eval_net/l1/b1/Adam/AssignAssign"C_train/Critic/eval_net/l1/b1/Adam4C_train/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
™
'C_train/Critic/eval_net/l1/b1/Adam/readIdentity"C_train/Critic/eval_net/l1/b1/Adam*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes

:
µ
6C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *(
_class
loc:@Critic/eval_net/l1/b1
¬
$C_train/Critic/eval_net/l1/b1/Adam_1
VariableV2*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
П
+C_train/Critic/eval_net/l1/b1/Adam_1/AssignAssign$C_train/Critic/eval_net/l1/b1/Adam_16C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
Ѓ
)C_train/Critic/eval_net/l1/b1/Adam_1/readIdentity$C_train/Critic/eval_net/l1/b1/Adam_1*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes

:
≈
=C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
“
+C_train/Critic/eval_net/q/dense/kernel/Adam
VariableV2*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
≠
2C_train/Critic/eval_net/q/dense/kernel/Adam/AssignAssign+C_train/Critic/eval_net/q/dense/kernel/Adam=C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
≈
0C_train/Critic/eval_net/q/dense/kernel/Adam/readIdentity+C_train/Critic/eval_net/q/dense/kernel/Adam*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
«
?C_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
‘
-C_train/Critic/eval_net/q/dense/kernel/Adam_1
VariableV2*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
≥
4C_train/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign-C_train/Critic/eval_net/q/dense/kernel/Adam_1?C_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
…
2C_train/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity-C_train/Critic/eval_net/q/dense/kernel/Adam_1*
_output_shapes

:*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
є
;C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
valueB*    */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
∆
)C_train/Critic/eval_net/q/dense/bias/Adam
VariableV2*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
°
0C_train/Critic/eval_net/q/dense/bias/Adam/AssignAssign)C_train/Critic/eval_net/q/dense/bias/Adam;C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
ї
.C_train/Critic/eval_net/q/dense/bias/Adam/readIdentity)C_train/Critic/eval_net/q/dense/bias/Adam*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
ї
=C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
»
+C_train/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
І
2C_train/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign+C_train/Critic/eval_net/q/dense/bias/Adam_1=C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
њ
0C_train/Critic/eval_net/q/dense/bias/Adam_1/readIdentity+C_train/Critic/eval_net/q/dense/bias/Adam_1*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
_
C_train/Adam/learning_rateConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
W
C_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
W
C_train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wЊ?
Y
C_train/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
А
5C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdamCritic/eval_net/l1/w1_s$C_train/Critic/eval_net/l1/w1_s/Adam&C_train/Critic/eval_net/l1/w1_s/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonKC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:P
В
5C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdamCritic/eval_net/l1/w1_a$C_train/Critic/eval_net/l1/w1_a/Adam&C_train/Critic/eval_net/l1/w1_a/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonMC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
х
3C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam	ApplyAdamCritic/eval_net/l1/b1"C_train/Critic/eval_net/l1/b1/Adam$C_train/Critic/eval_net/l1/b1/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonJC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*(
_class
loc:@Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
®
<C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdamCritic/eval_net/q/dense/kernel+C_train/Critic/eval_net/q/dense/kernel/Adam-C_train/Critic/eval_net/q/dense/kernel/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonPC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:
Ы
:C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdamCritic/eval_net/q/dense/bias)C_train/Critic/eval_net/q/dense/bias/Adam+C_train/Critic/eval_net/q/dense/bias/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonQC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
≤
C_train/Adam/mulMulC_train/beta1_power/readC_train/Adam/beta14^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
Є
C_train/Adam/AssignAssignC_train/beta1_powerC_train/Adam/mul*
_output_shapes
: *
use_locking( *
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(
і
C_train/Adam/mul_1MulC_train/beta2_power/readC_train/Adam/beta24^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
Љ
C_train/Adam/Assign_1AssignC_train/beta2_powerC_train/Adam/mul_1*
use_locking( *
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
д
C_train/AdamNoOp^C_train/Adam/Assign^C_train/Adam/Assign_14^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam
u
a_grad/gradients/ShapeShapeCritic/eval_net/q/dense/BiasAdd*
out_type0*
_output_shapes
:*
T0
_
a_grad/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Х
a_grad/gradients/FillFilla_grad/gradients/Shapea_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
£
Aa_grad/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrada_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
ў
;a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMula_grad/gradients/Fill#Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(
∆
=a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMulCritic/eval_net/l1/Relua_grad/gradients/Fill*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
 
6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad;a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulCritic/eval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€
К
4a_grad/gradients/Critic/eval_net/l1/add_1_grad/ShapeShapeCritic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
З
6a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
И
Da_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape6a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ч
2a_grad/gradients/Critic/eval_net/l1/add_1_grad/SumSum6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradDa_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
л
6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeReshape2a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ы
4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum_1Sum6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradFa_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
и
8a_grad/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum_16a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
Л
2a_grad/gradients/Critic/eval_net/l1/add_grad/ShapeShapeCritic/eval_net/l1/MatMul*
out_type0*
_output_shapes
:*
T0
П
4a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1ShapeCritic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
В
Ba_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs2a_grad/gradients/Critic/eval_net/l1/add_grad/Shape4a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
у
0a_grad/gradients/Critic/eval_net/l1/add_grad/SumSum6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeBa_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
е
4a_grad/gradients/Critic/eval_net/l1/add_grad/ReshapeReshape0a_grad/gradients/Critic/eval_net/l1/add_grad/Sum2a_grad/gradients/Critic/eval_net/l1/add_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
ч
2a_grad/gradients/Critic/eval_net/l1/add_grad/Sum_1Sum6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeDa_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
л
6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1Reshape2a_grad/gradients/Critic/eval_net/l1/add_grad/Sum_14a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
р
8a_grad/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(
а
:a_grad/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMulCritic/StopGradient6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
L
mul_8/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
^
mul_8Mulmul_8/xCritic/target_net/l1/w1_s/read*
T0*
_output_shapes

:P
L
mul_9/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
\
mul_9Mulmul_9/xCritic/eval_net/l1/w1_s/read*
_output_shapes

:P*
T0
C
add_4Addmul_8mul_9*
_output_shapes

:P*
T0
і
Assign_4AssignCritic/target_net/l1/w1_sadd_4*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
M
mul_10/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
`
mul_10Mulmul_10/xCritic/target_net/l1/w1_a/read*
T0*
_output_shapes

:
M
mul_11/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
^
mul_11Mulmul_11/xCritic/eval_net/l1/w1_a/read*
T0*
_output_shapes

:
E
add_5Addmul_10mul_11*
_output_shapes

:*
T0
і
Assign_5AssignCritic/target_net/l1/w1_aadd_5*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
validate_shape(
M
mul_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
^
mul_12Mulmul_12/xCritic/target_net/l1/b1/read*
_output_shapes

:*
T0
M
mul_13/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
\
mul_13Mulmul_13/xCritic/eval_net/l1/b1/read*
T0*
_output_shapes

:
E
add_6Addmul_12mul_13*
T0*
_output_shapes

:
∞
Assign_6AssignCritic/target_net/l1/b1add_6*
use_locking(*
T0**
_class 
loc:@Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
M
mul_14/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
g
mul_14Mulmul_14/x%Critic/target_net/q/dense/kernel/read*
T0*
_output_shapes

:
M
mul_15/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
e
mul_15Mulmul_15/x#Critic/eval_net/q/dense/kernel/read*
_output_shapes

:*
T0
E
add_7Addmul_14mul_15*
_output_shapes

:*
T0
¬
Assign_7Assign Critic/target_net/q/dense/kerneladd_7*
use_locking(*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
M
mul_16/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
a
mul_16Mulmul_16/x#Critic/target_net/q/dense/bias/read*
T0*
_output_shapes
:
M
mul_17/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
_
mul_17Mulmul_17/x!Critic/eval_net/q/dense/bias/read*
T0*
_output_shapes
:
A
add_8Addmul_16mul_17*
_output_shapes
:*
T0
Ї
Assign_8AssignCritic/target_net/q/dense/biasadd_8*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
u
policy_grads/gradients/ShapeShapeActor/eval_net/a/scaled_a*
_output_shapes
:*
T0*
out_type0
e
 policy_grads/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  А?*
dtype0
І
policy_grads/gradients/FillFillpolicy_grads/gradients/Shape policy_grads/gradients/grad_ys_0*'
_output_shapes
:€€€€€€€€€*
T0*

index_type0
Т
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/ShapeShapeActor/eval_net/a/a/Tanh*
out_type0*
_output_shapes
:*
T0
А
=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Э
Kpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ђ
9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/MulMulpolicy_grads/gradients/FillActor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
И
9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/SumSum9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/MulKpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
А
=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/ReshapeReshape9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
™
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Mul_1MulActor/eval_net/a/a/Tanhpolicy_grads/gradients/Fill*'
_output_shapes
:€€€€€€€€€*
T0
О
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum_1Sum;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Mul_1Mpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
х
?policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum_1=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
“
<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGradTanhGradActor/eval_net/a/a/Tanh=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€
Ћ
Bpolicy_grads/gradients/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGrad<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:*
T0
ь
<policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMulMatMul<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGradActor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(
н
>policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMulActor/eval_net/l1/Relu<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGrad*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
ѕ
;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGradReluGrad<policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMulActor/eval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€
…
Apolicy_grads/gradients/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
щ
;policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMulMatMul;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGradActor/eval_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€P*
transpose_b(
Ў
=policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGrad*
T0*
transpose_a(*
_output_shapes

:P*
transpose_b( 
Т
!A_train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: **
_class 
loc:@Actor/eval_net/a/a/bias*
valueB
 *fff?
£
A_train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias
“
A_train/beta1_power/AssignAssignA_train/beta1_power!A_train/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
Ж
A_train/beta1_power/readIdentityA_train/beta1_power*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
Т
!A_train/beta2_power/initial_valueConst**
_class 
loc:@Actor/eval_net/a/a/bias*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
£
A_train/beta2_power
VariableV2*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
“
A_train/beta2_power/AssignAssignA_train/beta2_power!A_train/beta2_power/initial_value*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ж
A_train/beta2_power/readIdentityA_train/beta2_power**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
≈
GA_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"P      *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
ѓ
=A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *+
_class!
loc:@Actor/eval_net/l1/kernel
ѓ
7A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillGA_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensor=A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
∆
%A_train/Actor/eval_net/l1/kernel/Adam
VariableV2*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel
Х
,A_train/Actor/eval_net/l1/kernel/Adam/AssignAssign%A_train/Actor/eval_net/l1/kernel/Adam7A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:P*
use_locking(
≥
*A_train/Actor/eval_net/l1/kernel/Adam/readIdentity%A_train/Actor/eval_net/l1/kernel/Adam*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
«
IA_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"P      *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
±
?A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0
µ
9A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillIA_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensor?A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
»
'A_train/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P
Ы
.A_train/Actor/eval_net/l1/kernel/Adam_1/AssignAssign'A_train/Actor/eval_net/l1/kernel/Adam_19A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
Ј
,A_train/Actor/eval_net/l1/kernel/Adam_1/readIdentity'A_train/Actor/eval_net/l1/kernel/Adam_1*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
≠
5A_train/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *)
_class
loc:@Actor/eval_net/l1/bias
Ї
#A_train/Actor/eval_net/l1/bias/Adam
VariableV2*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Й
*A_train/Actor/eval_net/l1/bias/Adam/AssignAssign#A_train/Actor/eval_net/l1/bias/Adam5A_train/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
©
(A_train/Actor/eval_net/l1/bias/Adam/readIdentity#A_train/Actor/eval_net/l1/bias/Adam*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
_output_shapes
:
ѓ
7A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
valueB*    *)
_class
loc:@Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:
Љ
%A_train/Actor/eval_net/l1/bias/Adam_1
VariableV2*
_output_shapes
:*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:*
dtype0
П
,A_train/Actor/eval_net/l1/bias/Adam_1/AssignAssign%A_train/Actor/eval_net/l1/bias/Adam_17A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
≠
*A_train/Actor/eval_net/l1/bias/Adam_1/readIdentity%A_train/Actor/eval_net/l1/bias/Adam_1*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
_output_shapes
:
ї
8A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:
»
&A_train/Actor/eval_net/a/a/kernel/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@Actor/eval_net/a/a/kernel
Щ
-A_train/Actor/eval_net/a/a/kernel/Adam/AssignAssign&A_train/Actor/eval_net/a/a/kernel/Adam8A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(
ґ
+A_train/Actor/eval_net/a/a/kernel/Adam/readIdentity&A_train/Actor/eval_net/a/a/kernel/Adam*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
љ
:A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:
 
(A_train/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
Я
/A_train/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign(A_train/Actor/eval_net/a/a/kernel/Adam_1:A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
Ї
-A_train/Actor/eval_net/a/a/kernel/Adam_1/readIdentity(A_train/Actor/eval_net/a/a/kernel/Adam_1*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
ѓ
6A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
valueB*    **
_class 
loc:@Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
Љ
$A_train/Actor/eval_net/a/a/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:
Н
+A_train/Actor/eval_net/a/a/bias/Adam/AssignAssign$A_train/Actor/eval_net/a/a/bias/Adam6A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias
ђ
)A_train/Actor/eval_net/a/a/bias/Adam/readIdentity$A_train/Actor/eval_net/a/a/bias/Adam*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
:
±
8A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    **
_class 
loc:@Actor/eval_net/a/a/bias
Њ
&A_train/Actor/eval_net/a/a/bias/Adam_1
VariableV2**
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
У
-A_train/Actor/eval_net/a/a/bias/Adam_1/AssignAssign&A_train/Actor/eval_net/a/a/bias/Adam_18A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
∞
+A_train/Actor/eval_net/a/a/bias/Adam_1/readIdentity&A_train/Actor/eval_net/a/a/bias/Adam_1*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
:
_
A_train/Adam/learning_rateConst*
_output_shapes
: *
valueB
 *oГЇ*
dtype0
W
A_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
W
A_train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wЊ?
Y
A_train/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
ч
6A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdamActor/eval_net/l1/kernel%A_train/Actor/eval_net/l1/kernel/Adam'A_train/Actor/eval_net/l1/kernel/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilon=policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_locking( *
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes

:P
н
4A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam	ApplyAdamActor/eval_net/l1/bias#A_train/Actor/eval_net/l1/bias/Adam%A_train/Actor/eval_net/l1/bias/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilonApolicy_grads/gradients/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
use_locking( *
T0*)
_class
loc:@Actor/eval_net/l1/bias*
use_nesterov( 
э
7A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdamActor/eval_net/a/a/kernel&A_train/Actor/eval_net/a/a/kernel/Adam(A_train/Actor/eval_net/a/a/kernel/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilon>policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
_output_shapes

:*
use_locking( *
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
use_nesterov( 
у
5A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdamActor/eval_net/a/a/bias$A_train/Actor/eval_net/a/a/bias/Adam&A_train/Actor/eval_net/a/a/bias/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilonBpolicy_grads/gradients/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
ф
A_train/Adam/mulMulA_train/beta1_power/readA_train/Adam/beta16^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
Ї
A_train/Adam/AssignAssignA_train/beta1_powerA_train/Adam/mul*
use_locking( *
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
ц
A_train/Adam/mul_1MulA_train/beta2_power/readA_train/Adam/beta26^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
Њ
A_train/Adam/Assign_1AssignA_train/beta2_powerA_train/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
§
A_train/AdamNoOp^A_train/Adam/Assign^A_train/Adam/Assign_16^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam
в
initNoOp,^A_train/Actor/eval_net/a/a/bias/Adam/Assign.^A_train/Actor/eval_net/a/a/bias/Adam_1/Assign.^A_train/Actor/eval_net/a/a/kernel/Adam/Assign0^A_train/Actor/eval_net/a/a/kernel/Adam_1/Assign+^A_train/Actor/eval_net/l1/bias/Adam/Assign-^A_train/Actor/eval_net/l1/bias/Adam_1/Assign-^A_train/Actor/eval_net/l1/kernel/Adam/Assign/^A_train/Actor/eval_net/l1/kernel/Adam_1/Assign^A_train/beta1_power/Assign^A_train/beta2_power/Assign^Actor/eval_net/a/a/bias/Assign!^Actor/eval_net/a/a/kernel/Assign^Actor/eval_net/l1/bias/Assign ^Actor/eval_net/l1/kernel/Assign!^Actor/target_net/a/a/bias/Assign#^Actor/target_net/a/a/kernel/Assign ^Actor/target_net/l1/bias/Assign"^Actor/target_net/l1/kernel/Assign*^C_train/Critic/eval_net/l1/b1/Adam/Assign,^C_train/Critic/eval_net/l1/b1/Adam_1/Assign,^C_train/Critic/eval_net/l1/w1_a/Adam/Assign.^C_train/Critic/eval_net/l1/w1_a/Adam_1/Assign,^C_train/Critic/eval_net/l1/w1_s/Adam/Assign.^C_train/Critic/eval_net/l1/w1_s/Adam_1/Assign1^C_train/Critic/eval_net/q/dense/bias/Adam/Assign3^C_train/Critic/eval_net/q/dense/bias/Adam_1/Assign3^C_train/Critic/eval_net/q/dense/kernel/Adam/Assign5^C_train/Critic/eval_net/q/dense/kernel/Adam_1/Assign^C_train/beta1_power/Assign^C_train/beta2_power/Assign^Critic/eval_net/l1/b1/Assign^Critic/eval_net/l1/w1_a/Assign^Critic/eval_net/l1/w1_s/Assign$^Critic/eval_net/q/dense/bias/Assign&^Critic/eval_net/q/dense/kernel/Assign^Critic/target_net/l1/b1/Assign!^Critic/target_net/l1/w1_a/Assign!^Critic/target_net/l1/w1_s/Assign&^Critic/target_net/q/dense/bias/Assign(^Critic/target_net/q/dense/kernel/Assign"&"и

trainable_variables–
Ќ

Ц
Actor/eval_net/l1/kernel:0Actor/eval_net/l1/kernel/AssignActor/eval_net/l1/kernel/read:024Actor/eval_net/l1/kernel/Initializer/random_normal:08
Ж
Actor/eval_net/l1/bias:0Actor/eval_net/l1/bias/AssignActor/eval_net/l1/bias/read:02*Actor/eval_net/l1/bias/Initializer/Const:08
Ъ
Actor/eval_net/a/a/kernel:0 Actor/eval_net/a/a/kernel/Assign Actor/eval_net/a/a/kernel/read:025Actor/eval_net/a/a/kernel/Initializer/random_normal:08
К
Actor/eval_net/a/a/bias:0Actor/eval_net/a/a/bias/AssignActor/eval_net/a/a/bias/read:02+Actor/eval_net/a/a/bias/Initializer/Const:08
Т
Critic/eval_net/l1/w1_s:0Critic/eval_net/l1/w1_s/AssignCritic/eval_net/l1/w1_s/read:023Critic/eval_net/l1/w1_s/Initializer/random_normal:08
Т
Critic/eval_net/l1/w1_a:0Critic/eval_net/l1/w1_a/AssignCritic/eval_net/l1/w1_a/read:023Critic/eval_net/l1/w1_a/Initializer/random_normal:08
В
Critic/eval_net/l1/b1:0Critic/eval_net/l1/b1/AssignCritic/eval_net/l1/b1/read:02)Critic/eval_net/l1/b1/Initializer/Const:08
Ѓ
 Critic/eval_net/q/dense/kernel:0%Critic/eval_net/q/dense/kernel/Assign%Critic/eval_net/q/dense/kernel/read:02:Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
Ю
Critic/eval_net/q/dense/bias:0#Critic/eval_net/q/dense/bias/Assign#Critic/eval_net/q/dense/bias/read:020Critic/eval_net/q/dense/bias/Initializer/Const:08"*
train_op

C_train/Adam
A_train/Adam"ѕ5
	variablesЅ5Њ5
Ц
Actor/eval_net/l1/kernel:0Actor/eval_net/l1/kernel/AssignActor/eval_net/l1/kernel/read:024Actor/eval_net/l1/kernel/Initializer/random_normal:08
Ж
Actor/eval_net/l1/bias:0Actor/eval_net/l1/bias/AssignActor/eval_net/l1/bias/read:02*Actor/eval_net/l1/bias/Initializer/Const:08
Ъ
Actor/eval_net/a/a/kernel:0 Actor/eval_net/a/a/kernel/Assign Actor/eval_net/a/a/kernel/read:025Actor/eval_net/a/a/kernel/Initializer/random_normal:08
К
Actor/eval_net/a/a/bias:0Actor/eval_net/a/a/bias/AssignActor/eval_net/a/a/bias/read:02+Actor/eval_net/a/a/bias/Initializer/Const:08
Ь
Actor/target_net/l1/kernel:0!Actor/target_net/l1/kernel/Assign!Actor/target_net/l1/kernel/read:026Actor/target_net/l1/kernel/Initializer/random_normal:0
М
Actor/target_net/l1/bias:0Actor/target_net/l1/bias/AssignActor/target_net/l1/bias/read:02,Actor/target_net/l1/bias/Initializer/Const:0
†
Actor/target_net/a/a/kernel:0"Actor/target_net/a/a/kernel/Assign"Actor/target_net/a/a/kernel/read:027Actor/target_net/a/a/kernel/Initializer/random_normal:0
Р
Actor/target_net/a/a/bias:0 Actor/target_net/a/a/bias/Assign Actor/target_net/a/a/bias/read:02-Actor/target_net/a/a/bias/Initializer/Const:0
Т
Critic/eval_net/l1/w1_s:0Critic/eval_net/l1/w1_s/AssignCritic/eval_net/l1/w1_s/read:023Critic/eval_net/l1/w1_s/Initializer/random_normal:08
Т
Critic/eval_net/l1/w1_a:0Critic/eval_net/l1/w1_a/AssignCritic/eval_net/l1/w1_a/read:023Critic/eval_net/l1/w1_a/Initializer/random_normal:08
В
Critic/eval_net/l1/b1:0Critic/eval_net/l1/b1/AssignCritic/eval_net/l1/b1/read:02)Critic/eval_net/l1/b1/Initializer/Const:08
Ѓ
 Critic/eval_net/q/dense/kernel:0%Critic/eval_net/q/dense/kernel/Assign%Critic/eval_net/q/dense/kernel/read:02:Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
Ю
Critic/eval_net/q/dense/bias:0#Critic/eval_net/q/dense/bias/Assign#Critic/eval_net/q/dense/bias/read:020Critic/eval_net/q/dense/bias/Initializer/Const:08
Ш
Critic/target_net/l1/w1_s:0 Critic/target_net/l1/w1_s/Assign Critic/target_net/l1/w1_s/read:025Critic/target_net/l1/w1_s/Initializer/random_normal:0
Ш
Critic/target_net/l1/w1_a:0 Critic/target_net/l1/w1_a/Assign Critic/target_net/l1/w1_a/read:025Critic/target_net/l1/w1_a/Initializer/random_normal:0
И
Critic/target_net/l1/b1:0Critic/target_net/l1/b1/AssignCritic/target_net/l1/b1/read:02+Critic/target_net/l1/b1/Initializer/Const:0
і
"Critic/target_net/q/dense/kernel:0'Critic/target_net/q/dense/kernel/Assign'Critic/target_net/q/dense/kernel/read:02<Critic/target_net/q/dense/kernel/Initializer/random_normal:0
§
 Critic/target_net/q/dense/bias:0%Critic/target_net/q/dense/bias/Assign%Critic/target_net/q/dense/bias/read:022Critic/target_net/q/dense/bias/Initializer/Const:0
t
C_train/beta1_power:0C_train/beta1_power/AssignC_train/beta1_power/read:02#C_train/beta1_power/initial_value:0
t
C_train/beta2_power:0C_train/beta2_power/AssignC_train/beta2_power/read:02#C_train/beta2_power/initial_value:0
Љ
&C_train/Critic/eval_net/l1/w1_s/Adam:0+C_train/Critic/eval_net/l1/w1_s/Adam/Assign+C_train/Critic/eval_net/l1/w1_s/Adam/read:028C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros:0
ƒ
(C_train/Critic/eval_net/l1/w1_s/Adam_1:0-C_train/Critic/eval_net/l1/w1_s/Adam_1/Assign-C_train/Critic/eval_net/l1/w1_s/Adam_1/read:02:C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros:0
Љ
&C_train/Critic/eval_net/l1/w1_a/Adam:0+C_train/Critic/eval_net/l1/w1_a/Adam/Assign+C_train/Critic/eval_net/l1/w1_a/Adam/read:028C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros:0
ƒ
(C_train/Critic/eval_net/l1/w1_a/Adam_1:0-C_train/Critic/eval_net/l1/w1_a/Adam_1/Assign-C_train/Critic/eval_net/l1/w1_a/Adam_1/read:02:C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros:0
і
$C_train/Critic/eval_net/l1/b1/Adam:0)C_train/Critic/eval_net/l1/b1/Adam/Assign)C_train/Critic/eval_net/l1/b1/Adam/read:026C_train/Critic/eval_net/l1/b1/Adam/Initializer/zeros:0
Љ
&C_train/Critic/eval_net/l1/b1/Adam_1:0+C_train/Critic/eval_net/l1/b1/Adam_1/Assign+C_train/Critic/eval_net/l1/b1/Adam_1/read:028C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros:0
Ў
-C_train/Critic/eval_net/q/dense/kernel/Adam:02C_train/Critic/eval_net/q/dense/kernel/Adam/Assign2C_train/Critic/eval_net/q/dense/kernel/Adam/read:02?C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros:0
а
/C_train/Critic/eval_net/q/dense/kernel/Adam_1:04C_train/Critic/eval_net/q/dense/kernel/Adam_1/Assign4C_train/Critic/eval_net/q/dense/kernel/Adam_1/read:02AC_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros:0
–
+C_train/Critic/eval_net/q/dense/bias/Adam:00C_train/Critic/eval_net/q/dense/bias/Adam/Assign0C_train/Critic/eval_net/q/dense/bias/Adam/read:02=C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros:0
Ў
-C_train/Critic/eval_net/q/dense/bias/Adam_1:02C_train/Critic/eval_net/q/dense/bias/Adam_1/Assign2C_train/Critic/eval_net/q/dense/bias/Adam_1/read:02?C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros:0
t
A_train/beta1_power:0A_train/beta1_power/AssignA_train/beta1_power/read:02#A_train/beta1_power/initial_value:0
t
A_train/beta2_power:0A_train/beta2_power/AssignA_train/beta2_power/read:02#A_train/beta2_power/initial_value:0
ј
'A_train/Actor/eval_net/l1/kernel/Adam:0,A_train/Actor/eval_net/l1/kernel/Adam/Assign,A_train/Actor/eval_net/l1/kernel/Adam/read:029A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros:0
»
)A_train/Actor/eval_net/l1/kernel/Adam_1:0.A_train/Actor/eval_net/l1/kernel/Adam_1/Assign.A_train/Actor/eval_net/l1/kernel/Adam_1/read:02;A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros:0
Є
%A_train/Actor/eval_net/l1/bias/Adam:0*A_train/Actor/eval_net/l1/bias/Adam/Assign*A_train/Actor/eval_net/l1/bias/Adam/read:027A_train/Actor/eval_net/l1/bias/Adam/Initializer/zeros:0
ј
'A_train/Actor/eval_net/l1/bias/Adam_1:0,A_train/Actor/eval_net/l1/bias/Adam_1/Assign,A_train/Actor/eval_net/l1/bias/Adam_1/read:029A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros:0
ƒ
(A_train/Actor/eval_net/a/a/kernel/Adam:0-A_train/Actor/eval_net/a/a/kernel/Adam/Assign-A_train/Actor/eval_net/a/a/kernel/Adam/read:02:A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros:0
ћ
*A_train/Actor/eval_net/a/a/kernel/Adam_1:0/A_train/Actor/eval_net/a/a/kernel/Adam_1/Assign/A_train/Actor/eval_net/a/a/kernel/Adam_1/read:02<A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros:0
Љ
&A_train/Actor/eval_net/a/a/bias/Adam:0+A_train/Actor/eval_net/a/a/bias/Adam/Assign+A_train/Actor/eval_net/a/a/bias/Adam/read:028A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zeros:0
ƒ
(A_train/Actor/eval_net/a/a/bias/Adam_1:0-A_train/Actor/eval_net/a/a/bias/Adam_1/Assign-A_train/Actor/eval_net/a/a/bias/Adam_1/read:02:A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros:0rїУR