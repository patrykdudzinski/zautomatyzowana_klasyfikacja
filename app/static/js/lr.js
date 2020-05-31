//Przywróć defaultowy stan formularza
function default_lr_state(){
    $('#form_lr_verbose').addClass('hidden')
    $('#form_lr_l1_ratio').addClass('hidden')
    $('#lr_wrapper select option').prop('disabled', false)
    $('#form_lr_warm_start').prop('disabled', false)
}


// JAVASCRIPT DO REGRESJI LOGISTYCZNEJ //
$(function(){
    $('#form_lr_algortithm').on('change', function(){
        default_lr_state();
        switch($(this).find('option:selected').val()){
            case 'newton-cg':
                $('#form_lr_penalty option[value="l1"]').prop('disabled', true)  
                $('#form_lr_penalty option[value="elasticnet"]').prop('disabled', true)
                $('#form_lr_dual option[value="1"]').prop('disabled', true)
                break;
            case 'liblinear':
                $('#form_lr_multi_class option[value="multinomial"]').prop('disabled', true) 
                $('#form_lr_penalty option[value="none"]').prop('disabled', true) 
                $('#form_lr_verbose').removeClass('hidden')
                $('#form_lr_warm_start').prop('disabled', true)
                $('#form_lr_penalty option[value="elasticnet"]').prop('disabled', true)
                break;
            case 'lbfgs':
                $('#form_lr_penalty option[value="l1"]').prop('disabled', true)  
                $('#form_lr_penalty option[value="elasticnet"]').prop('disabled', true)
                break
                case 'sag':
                $('#form_lr_penalty option[value="elasticnet"]').prop('disabled', true)
                $('#form_lr_penalty option[value="l1"]').prop('disabled', true)
                $('#form_lr_dual option[value="1"]').prop('disabled', true)
            default:
                break;
        }
    })    
    $('#form_lr_penalty').on('change', function(){
        default_lr_state();
        switch($(this).find('option:selected').val()){
            case 'elasticnet':
                $('#form_lr_l1_ratio').removeClass('hidden') 
                break;
            default:
                break;
        }
    })
})