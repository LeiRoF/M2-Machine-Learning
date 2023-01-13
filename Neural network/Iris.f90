program OR_gate

    ! MARCHE PO :/

    implicit none

    ! Constants ---------------------------------------------------------------

    integer, parameter :: N = 5   ! Number of inputs
    integer, parameter :: nodes = 3 ! Number of neurones
    real               :: a = 0.1 ! Learning rate

    ! Variables ---------------------------------------------------------------

    real,    allocatable                 :: X(:,:) ! Input
    integer, dimension(nodes)            :: bin, bin2 ! Binary representation of the output
    real,    dimension(N,nodes)          :: w  ! Weights
    integer                              :: y, t  ! Output
    integer                              :: i,j   ! Loop counter
    integer                              :: tmp_int, tmp_int2   ! Temporary variable
    real                                 :: tmp_real
    integer, dimension(nodes)            :: debug_bin
    real,    dimension(nodes+1, nodes+2) :: confusion = 0
    real                                 :: r     ! Random number
    integer                              :: nlines ! Number of lines in the file
    integer, allocatable                 :: output(:) ! Expected output
    integer                              :: s     ! Sum of correct answers

    ! Getting the data -------------------------------------------------------

    ! Count the number of lines in the file
    nlines = 0
    open(42, file = 'iris.txt')
    do
        read(42,*, END=10)
        nlines = nlines + 1
    end do  
    10 close (42)
    ! print *, "Number of lines:", nlines

    ! Allocate the arrays
    allocate(X(nlines,N))
    allocate(output(nlines))

    ! Read the file
    open(42, file="iris.txt", status="old", action="read")
    do i= 1, nlines
        read(42, *) X(i, :)
        output(i) = int(X(i, N)) ! result
        X(i, N) = -1 ! bias
    end do
    close(42)

    output = output + 1

    ! Normalizing the data ---------------------------------------------------

    ! do i=1, N-1
    !     tmp_real = maxval(abs(X(:,i)))
    !     X(:,i) = X(:,i) / tmp_real
    ! end do

    ! Initialize the weights --------------------------------------------------

    do j=1, nodes
        do i=1, N
            call random_number(w(i,j))
        end do
    end do

    ! Train ------------------------------------------------------------------
    
    do i=1, 1000

        ! Randomly choose a line
        call random_number(r)
        j = int(r * nlines + 1)

        ! Get the expected output
        t = output(j)
    
        ! Simulate each neuron
        do j=1, nodes
            bin(j) = neuron(X(j,:), w(:,j), N)
        end do

        call dec_to_bin(t, bin2, nodes)

        ! Adjust the weights
        do j=1, nodes
            call adjust_weights(X, w(:,j), N, bin(j), bin2(j), a)
        end do
        
        call bin_to_dec(bin, y, nodes)
        confusion(t, y) = confusion(t, y) + 1

        if (mod(i, 1) == 0) then
            print *, "--------------------------------------------------"
            print *, " TRAINING PHASE:", i
            print *, " "
            print *, bin
            print *, "Expected:", t, "Got:", y
            print *, " "
            call confusion_matrix(confusion, nodes)
            print *, " "
            do j=1, nodes
                print *, "Weights for node", j, ":" , w(:,j)
            end do
        end if

    end do

    ! Test the OR gate --------------------------------------------------------

    s = 0
    do i=1, 100

        ! Randomly choose a line
        call random_number(r)
        j = int(r * nlines + 1)
        t = output(j)

        ! For all neurons
        do j=1, nodes
            ! Simulate the neuron
            bin(j) = neuron(X(j,:), w(:,j), N)
        end do

        ! Converting back the binary sequence to and integer
        call bin_to_dec(bin, y, nodes)

        ! Computing true/false positive/negative
        confusion(t,y) = confusion(t,y) + 1

    end do

    ! Print the weights -------------------------------------------------------

    ! print *, "Weights:", w

    ! Print the confusion matrix ----------------------------------------------

    ! print *, "--------------------------------------------------"
    ! print *, " TEST PHASE"
    ! print *, " "
    ! call confusion_matrix(confusion, nodes)
    ! print *, " "





    ! Functions and subroutines ===============================================





    contains

        ! Compute accuracy and print confusion matrix -------------------------

        subroutine confusion_matrix(confusion, nodes)

            real, dimension(nodes+1, nodes+2) :: confusion
            real,    dimension(nodes)         :: precision, recall
            real                              :: accuracy, trace
            integer                           :: nodes
            integer                           :: k

            print *, "Confusion matrix:"

            trace = 0
            do k=1,nodes
                precision(k) = real(confusion(k,k)) / sum(confusion(:,k))
                recall(k) = real(confusion(k,k)) / sum(confusion(k,:))
                trace = trace + confusion(k,k)
            end do
            confusion(nodes+1,:) = recall
            confusion(:,nodes+2) = precision
            accuracy = real(trace) / sum(confusion)

            print *, "Expected\Got     1     2     3     Precision"
            do k=1,nodes+1
                if (k < nodes+1) then
                    print *, k, confusion(k,:)
                else
                    print *, "Recall:", confusion(k,:)
                end if
            end do

            ! print *, "Precision:", precision
            ! print *, "Recall:", recall

            ! print *, "Accuracy:", accuracy
        
        end subroutine confusion_matrix

        ! Neuron --------------------------------------------------------------

        function neuron(X, w, N)

            integer            :: N
            real, dimension(N) :: X
            real, dimension(N) :: w
            real               :: theta
            integer            :: neuron

            theta = weighted_sum(X, w, N)
            neuron = activation(theta)

        end function neuron

        ! Weighted sum --------------------------------------------------------

        function weighted_sum(X, w, N)
            
            integer               :: N
            real, dimension(N)    :: X
            real, dimension(N)    :: w
            real                  :: weighted_sum
            integer               :: i

            weighted_sum = 0
            do i = 1, N
                weighted_sum = weighted_sum + X(i) * w(i)
            end do

        end function weighted_sum

        ! Activation ----------------------------------------------------------

        function activation(theta)

            integer :: activation
            real    :: theta

            if (theta > 0) then
                activation = 1
            else
                activation = 0
            end if

        end function activation

        ! Adjust weights ------------------------------------------------------

        subroutine adjust_weights(X, w, N, y, t, a)

            integer               :: N
            real, dimension(N) :: X
            real, dimension(N)    :: w
            integer :: y, t, i
            real                  :: a

            do i = 1, N
                w(i) = w(i) - a * (y - t) * X(i)
            end do
        end subroutine adjust_weights

        ! Random bool ---------------------------------------------------------

        subroutine random_bool(r)
            integer, intent(out) :: r
            real ::tmp

            call random_number(tmp)
            r = int(tmp + 0.5)
        end subroutine random_bool

        ! Binary to decimal and vice versa ------------------------------------

        subroutine dec_to_bin(D, B, N)
                integer, intent(in)  :: D
                integer, intent(out) :: B(N)
                integer, intent(in)  :: N
                integer :: i, tmp
    
                tmp = D
                do i = 1, N
                    B(N-i+1) = mod(tmp, 2)
                    tmp = tmp / 2
                end do

                ! B = 0
                ! B(D) = 1

        end subroutine dec_to_bin

        subroutine bin_to_dec(B, D, N)
                integer, intent(in)  :: B(N)
                integer, intent(out) :: D
                integer, intent(in)  :: N
                integer :: i
    
                D = 0
                do i = 1, N
                    D = D + B(i) * 2**(N-i)
                end do

                if (D .eq. 3) then
                    D = 4
                else if (D > 4) then
                    D = 4
                else if (D .eq. 0) then
                    D = 4
                else if (D .eq. 4) then
                    D = 3
                end if

                ! do i = 1, N
                !     if (B(i) == 1) then
                !         D = i
                !     end if
                ! end do
        end subroutine bin_to_dec

end program OR_gate