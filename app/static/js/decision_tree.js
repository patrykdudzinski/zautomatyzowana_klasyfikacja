//JS do zarządzania formularzem drzewa decyzyjnego//
$(function(){
    $('#form_bt_max_features').on('change', function(){
        $('#max_features_float_wrapper').addClass('hidden')
        switch($(this).find('option:selected').val()){
            case 'user':
                $('#max_features_float_wrapper').removeClass('hidden') 
                break;
            default:
                break;
        }
    })
    
    $('#form_bt_weight_fraction').on('change', function(){
        var weight_fraction = parseFloat($(this).val());
        if(weight_fraction > 0.5 || weight_fraction < 0){
            $(this).addClass("is-invalid-input")
            $("#error_message").text("Niedozwolona wartość - minimalna waga liścia pomiędzy 0 a 0.5");
        }
    })
    
    $('#form_bt_max_features_float').on('change', function(){
        var max_features_value = $(this).val();
        if(max_features_value >=  parseInt($('#class_names_length').val())){
             $(this).addClass("is-invalid-input")
             $("#error_message").text("Wartość nie może przekraczać liczby klas = "+$('#class_names_length').val());
        }
    })
})