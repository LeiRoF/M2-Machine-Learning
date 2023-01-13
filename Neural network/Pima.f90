program OR_gate

    implicit none

    ! Constants ---------------------------------------------------------------

    integer, parameter    :: N = 9   ! Number of inputs
    real                  :: a = 0.1 ! Learning rate

    ! Variables ---------------------------------------------------------------

    real, allocatable     :: X(:,:)  ! Input
    real, dimension(N)    :: w     ! Weights
    integer               :: y, t  ! Output
    integer               :: i,j   ! Loop counter
    integer               :: tmp   ! Temporary variable
    integer               :: TP=0, TN=0, FP=0, FN=0 ! True positive, true negative, false positive, false negative
    real                  :: r     ! Random number
    integer               :: nlines ! Number of lines in the file
    integer, allocatable  :: truth_table(:) ! Truth table
    integer               :: ios   ! Iostat
    real                  :: frac  ! Fraction of correct answers
    integer               :: s     ! Sum of correct answers

    ! Getting the data -------------------------------------------------------

    ! Count the number of lines in the file
    nlines = 0
    open(42, file = 'pima_data2.txt')
    do
        read(42,*, END=10)
        nlines = nlines + 1
    end do  
    10 close (42)
    ! print *, "Number of lines:", nlines

    ! Allocate the arrays
    allocate(X(nlines,N))
    allocate(truth_table(nlines))

    ! Read the file
    open(42, file="pima_data2.txt", status="old", action="read")
    do i= 1, nlines
        read(42, *) X(i, :)
        truth_table(i) = X(i, N) ! result
        X(i, N) = -1 ! bias
    end do
    close(42)

    ! Normalizing the data ---------------------------------------------------

    ! Grouping age by range of 10 years
    ! do i=1, nlines
    !     tmp = int(X(i,8) / 10)
    !     X(i,8) = tmp
    ! end do

    do i=1, N-1
        tmp = maxval(X(:,i))
        X(:,i) = X(:,i) / tmp
    end do

    ! Initialize the weights --------------------------------------------------

    do i=1, N
        call random_number(w(i))
    end do

    ! Train the OR gate -------------------------------------------------------

    ! Reading the file
    
    do i=1, 10000

        ! Randomly choose a line
        call random_number(r)
        j = int(r * nlines + 1)

        ! Get the expected output
        t = truth_table(j)

        ! Simulation the neuron
        y = neuron(X(j,:), w, N)

        ! Adjust the weights
        call adjust_weights(X, w, N, y, t, a)

    end do

    ! Test the OR gate --------------------------------------------------------

    s = 0
    do i=1, 1000

        ! Randomly choose a line
        call random_number(r)
        j = int(r * nlines + 1)

        ! Simulation the neuron
        y = neuron(X(j,:), w, N)

        ! call random_number(r)
        ! y = int(r+0.5)

        ! Print the result
        ! print *, "[TEST] Input (",j,"):", X(j,:)
        ! print *, "       Output:", y, "Expected:", truth_table(j)
        ! print *, " "

        if (y .eq. truth_table(j)) then
            if(y == 1) then
                TP = TP + 1
            else
                TN = TN + 1
            end if
        else
            if(y == 1) then
                FP = FP + 1
            else
                FN = FN + 1
            end if
        end if
    end do

    ! Print the weights -------------------------------------------------------

    ! print *, "Weights:", w

    print *, "Recall:", real(TP) / (TP + FN), "Precision:", real(TP) / (TP + FP), "Accuracy:", real((TP + TN)) / (TP + TN + FP + FN)

    ! Functions and subroutines ===============================================

    contains

        ! Neuron --------------------------------------------------------------

        function neuron(X, w, N)

            integer               :: N
            real, dimension(N) :: X
            real, dimension(N)    :: w
            integer :: neuron, theta

            theta = weighted_sum(X, w, N)
            neuron = activation(theta)

        end function neuron

        ! Weighted sum --------------------------------------------------------

        function weighted_sum(X, w, N)
            
            integer               :: N
            real, dimension(N) :: X
            real, dimension(N)    :: w
            integer :: weighted_sum, i

            weighted_sum = 0
            do i = 1, N
                weighted_sum = weighted_sum + X(i) * w(i)
            end do

        end function weighted_sum

        ! Activation ----------------------------------------------------------

        function activation(theta)

            integer :: activation, theta

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
        end subroutine dec_to_bin

        subroutine bin_to_dec(B, D, N)
                integer, intent(in)  :: B(N)
                integer, intent(out) :: D
                integer, intent(in)  :: N
                integer :: i, tmp
    
                D = 0
                do i = 1, N
                    D = D + B(i) * 2**(N-i)
                end do
        end subroutine bin_to_dec

end program OR_gate