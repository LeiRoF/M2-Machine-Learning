program MLP

    implicit none

    ! Declaration of variables -----------------------------------------------

    integer :: layers  ! number of layers
    integer :: X_size  ! size of input vectors
    integer :: T_size  ! size of output vectors
    integer :: N       ! number of vectors in the dataset
    integer :: e, i, j, l!, k ! loop variables
    integer :: epochs  ! number of epochs

    real,    allocatable :: X(:,:), T(:,:)       ! dataset vectors
    real,    allocatable :: Y(:,:)               ! output vectors of each layer
    real,    allocatable :: error(:,:,:)         ! error vectors of each neuron
    real,    allocatable :: prediction(:)        ! output of the network
    real                 :: output               ! output of a neuron
    real,    allocatable :: X_norm(:), T_norm(:) ! normalization factors
    real                 :: a=0.1                ! learning rate
    integer, allocatable :: neurons(:)           ! number of neurons in each layer
    integer, allocatable :: n_weights(:)         ! number of weights in each layer
    real,    allocatable :: weights(:,:,:)       ! weights of the network
    character(len=1024)  :: file                 ! file to read from

    ! Initialization ---------------------------------------------------------

    ! Get the number of layers
    layers = get_layers()                ! get the number of layers

    ! Get the number of neurons in each layer
    allocate(neurons(layers))            ! allocate the neurons array
    call get_neurons(neurons)            ! get the number of neurons in each layer

    ! Get the size of input and output vectors
    X_size = get_input_size()            ! get the size of input vector
    T_size = neurons(layers)             ! get the size of output vector
    allocate(X_norm(X_size))             ! allocate the normalization factors for the input vectors
    allocate(T_norm(T_size))             ! allocate the normalization factors for the output vectors

    ! Allocate the arrays
    allocate(Y(layers, maxval(neurons))) ! allocate the output vector of each layer
    allocate(prediction(T_size))          ! allocate the output of the network
    allocate(n_weights(layers))          ! allocate the n_weights array

    ! Define the number of weights for a neuron in each layer
    n_weights(1) = X_size                ! compute the number of weights for the neurons in the first layer
    do i = 2, size(neurons) + 1
        n_weights(i) = neurons(i-1)      ! compute the number of weights for the neurons in the other layers
    end do

    ! Get the file to read data from
    call get_file(file)

    ! Allocate the weights and error arrays (layer, neuron, weight)
    allocate(weights(layers, maxval(neurons), max(maxval(neurons), X_size)))
    allocate(error(layers, maxval(neurons), max(maxval(neurons), X_size)))

    ! Initialize the weights array
    call random_number(weights)

    ! Read the input and output vectors from the file
    call read_data(file, X, T, N, X_size, T_size) ! X, T, N <- file
    call normalize_data(X, T, N, X_size, T_size, X_norm, T_norm)

    ! Ask the user for the number of epochs
    epochs = get_epochs()

    ! Training ---------------------------------------------------------------

    call print_weights(weights, neurons, n_weights)

    do e = 1, epochs

        print *, "=================================================="
        print *, "Epoch ", e, "/", epochs

        ! Shuffle the dataset
        call shuffle(X, T)

        ! For each vector in the dataset
        do i = 1, N

            print *, "----------"

            ! Forward pass ----------------------------------------------------
            ! For each layer
            do l = 1, layers
                ! For each neuron in the layer
                do j = 1, neurons(l)
                    ! Compute the output of the neuron
                    call stimulate_neuron(                                    &
                        X(i,:), weights(l,j,:), n_weights(l), output          &
                    )
                    Y(l,j) = output
                end do
            end do

            prediction = Y(layers, 1:T_size)

            ! Backward pass ---------------------------------------------------

            ! Computing the error of the last layer
            do j = 1, neurons(layers)
                error(layers,j,:) = a * (Y(layers,j) - T(i,j)) * weights(layers,j,:)
            end do

            ! For each layer
            do l = layers, 1, -1
                if (l < layers) then
                    ! For each neuron in the layer
                    do j = 1, neurons(l)
                        ! Compute the error of the neuron
                        error(l,j,:) = sum(error(l+1,j,:))/n_weights(l+1) * weights(l,j,:)
                        weights(l,j,:) = weights(l,j,:) - error(l,j,:)
                    end do
                end if
            end do

            call print_weights(weights, neurons, n_weights)

            print *, X_norm * X(i,:), "|", T_norm * T(i,:), " -> ", T_norm * prediction
        end do
    end do

    ! Testing ----------------------------------------------------------------



    !==========================================================================
    ! FUNCTIONS AND SUBROUTINES
    !==========================================================================



    contains

        ! Ask the user how many layers the network should have ----------------
        function get_layers() result(layers)
            implicit none
            integer :: layers
            ! write(*,*) "How many layers should the network have?"
            ! read(*,*) layers
            layers = 2
        end function get_layers

        ! Ask the user how many neurons should be in each layer ---------------
        subroutine get_neurons(neurons) !-> neurons(:)
            implicit none
            integer, intent(inout) :: neurons(:)
            ! integer :: i
            ! do i = 1, size(neurons)
            !     write(*,*) "How many neurons should be in layer ", i, "?"
            !     if (i == size(neurons)) write(*,*) "/!\ define also the size of output vector /!\"
            !     read(*,*) neurons(i)
            ! end do
            ! neurons = neurons
            neurons = (/ 3, 1 /)
        end subroutine get_neurons

        ! Ask the user for the size of the input vector -----------------------
        function get_input_size() result(X_size)
            implicit none
            integer :: X_size
            ! write(*,*) "What is the size of the input vector?"
            ! read(*,*) X_size
            X_size = 8
            X_size = 2
        end function get_input_size

        ! Ask the user for the file to read from ------------------------------
        subroutine get_file(file)
            implicit none
            character(len=1024), intent(inout) :: file
            ! write(*,*) "What file should be read from?"
            ! read(*,*) file
            file = "pima_data.txt"
            file = "test.txt"
        end subroutine get_file

        ! Ask the user for the number of epochs -------------------------------
        function get_epochs() result(epochs)
            implicit none
            integer :: epochs
            ! write(*,*) "How many epochs should be performed?"
            ! read(*,*) epochs
            epochs = 3
        end function get_epochs

        ! Read the input and output vectors from the file ---------------------
        subroutine read_data(file, X, T, N, X_size, T_size)
            implicit none
            character(len=1024),  intent(in   ) :: file
            integer,              intent(in   ) :: X_size, T_size
            real,    allocatable, intent(inout) :: X(:,:), T(:,:)
            integer,              intent(inout) :: N
            real, dimension(X_size + T_size)    :: D

            N = 0
            open(42, file=file)
            do
                read(42,*, END=10)
                N = N + 1
            end do  
            10 close (42)
            ! print *, "Number of lines:", nlines

            ! Allocate the arrays
            allocate(X(N, X_size))
            allocate(T(N, T_size))

            ! Read the file
            open(42, file=file, status="old", action="read")
            do i= 1, N
                read(42, *) D
                ! Fill the input and output vectors
                X(i,:) = D(1:X_size)
                T(i,:) = D(X_size+1:X_size+T_size)
            end do
            close(42)
        end subroutine read_data

        ! Normalize the input vectors -----------------------------------------
        subroutine normalize_data(X, T, N, X_size, T_size, X_norm, T_norm)
            implicit none
            real, intent(inout) :: X(:,:), T(:,:)
            integer, intent(in) :: X_size, T_size, N
            integer :: i
            real, intent(out) :: X_norm(:), T_norm(:)

            do i=1,X_size
                X_norm(i) = maxval(X(:,i))
            end do

            do i=1,T_size
                T_norm(i) = maxval(T(:,i))
            end do

            do i=1,N
                X(i,:) = X(i,:)/X_norm
                T(i,:) = T(i,:)/T_norm
            end do

        end subroutine normalize_data
        
        ! Shuffle the dataset --------------------------------------------------
        subroutine shuffle(X, T)
            implicit none
            real, intent(inout) :: X(:,:), T(:,:)
            integer :: i, j, k
            real :: tmp, r
            do i = 1, size(X, 1)
                call random_number(r)
                j = 1 + int(r * (size(X, 1) - 1))

                ! Swap X(i, :) and X(j, :)
                do k = 1, size(X, 2)
                    tmp = X(i, k)
                    X(i, k) = X(j, k)
                    X(j, k) = tmp
                end do
                ! Swap T(i, :) and T(j, :)
                do k = 1, size(T, 2)
                    tmp = T(i, k)
                    T(i, k) = T(j, k)
                    T(j, k) = tmp
                end do
            end do
        end subroutine shuffle
        
        ! Weighted sum ---------------------------------------------------------
        subroutine weighted_sum(X, weights, n_weights, s)
            implicit none
            real,    intent(in   ) :: X(:)
            real,    intent(in   ) :: weights(:)
            integer, intent(in   ) :: n_weights
            real,    intent(  out) :: s
            integer :: i
            s = 0
            do i = 1, n_weights
                s = s + X(i) * weights(i)
            end do
        end subroutine weighted_sum

        ! Activation function --------------------------------------------------
        subroutine activation(x, a)
            implicit none
            real, intent(in   ) :: x
            real, intent(  out) :: a
            a = 1 / (1 + exp(-x)) * 4
        end subroutine activation

        ! Compute the output of the neuron -------------------------------------
        subroutine stimulate_neuron(X, weights, n_weights, output)
            implicit none
            real,    intent(in   ) :: X(:)
            real,    intent(in   ) :: weights(:)
            integer, intent(in   ) :: n_weights
            real,    intent(  out) :: output
            real :: tmp
            call weighted_sum(X, weights, n_weights, tmp)
            call activation(tmp, output)
        end subroutine stimulate_neuron

        ! Print all the weights -----------------------------------------------
        subroutine print_weights(weights, neurons, n_weights)
            implicit none
            real, intent(in) :: weights(:,:,:)
            integer, intent(in) :: neurons(:), n_weights(:)
            integer :: i, j

            do i = 1, size(neurons)
                write(*,*) "Layer ", i
                do j = 1, neurons(i)
                    write(*,*) "    Neuron ", j, ":", weights(i, j, 1:n_weights(i))
                end do
            end do
        end subroutine print_weights
        
end program MLP