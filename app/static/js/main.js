function prepareMlpForm(){
    $('#shuffle_wrapper').addClass('hidden');
    $('#form_mlp_learning_rate_init').addClass('hidden');
    $('#form_mlp_epochs').addClass('hidden');    
    $('#mlp_learning_rate_wrapper').addClass('hidden');    
    $('#mlp_adam_wrapper').addClass('hidden');    
    $('#lbfgs_wrapper').addClass('hidden');    
}

function error_controller(error_code){
    switch(error_code){
        case '1':
            $('#error_message').text('Brak atrybutu class w zestawie danych')
            break;
        case '2':
            $('div#chart').hide();
            $('#error_message').text('Nieznany błąd. Spróbuj ponownie zmieniając parametry.')
        case '3':
            $('#error_message').text('Nieznany błąd. Spróbuj ponownie lub sprawdź zawartość pliku.')
        default:
            break;
    }    
}


$('document').ready(function(){
    $('#error_message').text('')
    if($('#parameters_filename').text() !== ""){
        if($('#code').val() === '0'){
            $('#parameters_list').show();
        }
        else{
            error_controller($('#code').val());
        }
    }
    
    $(document).foundation()
     
    /* kontroler do selectboxa z metodami uczenia 
       0 - LogisticRegression
       1 - kNN
       2 - Naiwny Bayes
       3 - SVC
       4 - Drzewo binarne
       5 - MLP
    */
    $('#form_method').on('change', function(){ 
        var selected_method = $(this).find('option:selected').val();
        prepareMlpForm()
        $('.additional_parameters.visible').removeClass('visible')
        switch(selected_method){
            case '0':
                $('#lr_wrapper').addClass('visible');
                break;
            case '1':
                $('#knn_wrapper').addClass('visible');
                break;            
            case '2':
                $('#nb_wrapper').addClass('visible');
                break;            
            case '3':
                $('#svc_wrapper').addClass('visible');
                break;            
            case '4':
                $('#bt_wrapper').addClass('visible');
                break;           
            case '5':
                $('#mlp_wrapper').addClass('visible');
                /* zdarzenia dla solvera */
                $('#form_mlp_solver').on('change', function(){
                    prepareMlpForm()
                    var chosen_option = $(this).find('option:selected').val();
                    if(chosen_option === 'sgd' || chosen_option === 'adam'){
                        $('#shuffle_wrapper').removeClass('hidden');
                        $('#form_mlp_learning_rate_init').removeClass('hidden');
                        $('#form_mlp_epochs').removeClass('hidden');
                    }
                    if(chosen_option === 'sgd'){
                        $('#mlp_learning_rate_wrapper').removeClass('hidden')
                    }                    
                    if(chosen_option === 'adam'){
                        $('#mlp_adam_wrapper').removeClass('hidden')
                    }                    
                    if(chosen_option === 'lbfgs'){
                        $('#lbfgs_wrapper').removeClass('hidden')
                    }
                })
                break;
            default:
                break;
        }
    });


    $('#fileuploader').on('change', function(){ 
        $('#send_file').click();
    })

    $('#send_file').on('click', function(){
        setTimeout(function(){
            var file = $('#fileuploader').val().split('\\');
            $.ajax({
                url: "/_get_file_data/",
                data: {'file' : file[file.length-1]},
                type: "POST",
                success: function(resp){
                    $('#parameters_list').show();
                    $('div#parameters_list').html(resp.data);
                }
            });
        }, 500)
    })    

    /*
        desc: Wywołanie formularza - sprawdzane są zależności pomiędzy wartościami hiperparametrów oraz przygotowywany json z wartościami.
        param: pola z formularza .form-option
        param: wybrana metoda klasyfikacji (wartość selecta form-method)
        return: JSON z danymi wyciągnietymi z kontrolera + wyświetlenie wykresu
    */
    $('#parameters_list #forecast_data_save').on('click', function(){
        $('#error_message').text('');
        $('.is-invalid-input').removeClass('is-invalid-input'); 
        var filename = $('#parameters_filename').text()+'.arff';
        var confirmation = confirm('Dokonać klasyfikacji dla zbioru '+filename+'?')
        if(confirmation === true){
            $.getScript('/static/js/check_parameters.js', function(){              
            var checker = check_parmeters($('#form_method option:selected').val());
                if(checker === true){
                    var hyperparams_array = [];
                    $('.form_option').each(function(key, element){
                        if(element.type === 'select-one'){
                            hyperparams_array.push($(this).attr('name')+":"+ $(this).find('option:selected').val())       
                        }
                        else if(element.type === 'checkbox'){
                            hyperparams_array.push($(this).attr('name')+":"+ $(this).prop('checked'))
                        }
                        else{
                            hyperparams_array.push($(this).attr('name')+":"+ $(this).val())       
                        }
                    })       

                    var hyperparams = JSON.stringify(hyperparams_array);
                    $.ajax({
                        url: "/_forecast_data/",
                        data: {'filename' : filename,
                               hyperparams: hyperparams
                        },
                        type: "POST",
                        success: function(resp){
                            $('div#chart').html(resp.data);
                            if($('#forecast_code').val() !== '0'){
                                error_controller($('#forecast_code').val());
                            }
                        }
                    });            
                }
            }); 
        }
    })
    
})