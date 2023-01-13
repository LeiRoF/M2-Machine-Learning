program OR_gate

    implicit none

    ! Constants ---------------------------------------------------------------

    integer, parameter    :: N = 4 ! Number of inputs
    real                  :: a=0.1 ! Learning rate
    integer, dimension(2**N) :: truth_table = [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1]

    ! Variables ---------------------------------------------------------------

    integer, dimension(N) :: X     ! Input
    real, dimension(N,N)  :: w1, w2 ! Weights for layer 1 and 2
    integer, dimension(N) :: y1,y2     ! Output of the first layer
    real, dimension(N)    :: w3    ! Weights for layer 3
    integer               :: y, t  ! Output
    integer               :: i,j   ! Loop counter
    integer               :: tmp   ! Temporary variable
    real                  :: r     ! Random number

    ! Initialize the weights --------------------------------------------------

    do i=1, N
        do j=1, N
            call random_number(w1(i,j))
            call random_number(w2(i,j))
        end do
        call random_number(w3(i))
    end do

    ! Train the OR gate -------------------------------------------------------

    do i=1, 10000
        ! Draw a random X from {0, 1}^N using uniform distribution
        do j=1, N
            call random_bool(X(j))
        end do

        do j=1, N
            ! Simulation the neuron
            y1(j) = neuron(X, w1(j,:), N)
        end do

        do j=1, N
            ! Simulation the neuron
            y2(j) = neuron(y1, w2(j,:), N)
        end do

        y = neuron(y2, w3, N)

        ! convert X to a number
        call bin_to_dec(X, tmp, N)

        ! Compute the target
        t = truth_table(tmp+1)

        ! Adjust the weights
        call adjust_weights(X, w1, w2, w3, N, y1, y2, y, t, a)
    end do

    ! Test the OR gate --------------------------------------------------------

    do i=1, N**2
        ! X = i in binary representation
        call dec_to_bin(i-1, X, N)

        ! Simulation the neural network
        do j=1, N
            y1(j) = neuron(X, w1(j,:), N)
        end do
        do j=1, N
            y2(j) = neuron(y1, w2(j,:), N)
        end do
        y = neuron(y2, w2, N)

        ! Print the result
        call bin_to_dec(X, tmp, N)
        print *, "Input:", X, "Output:", y, "Expected:", truth_table(tmp+1)

    end do

    ! Print the weights -------------------------------------------------------

    print *, "Weights for layer 1:"
    do i=1, N
        print *, w1(i,:)
    end do

    print *, "Weights for layer 2:"
    print *, w2

    ! Functions and subroutines ===============================================

    contains

        ! Neuron --------------------------------------------------------------

        function neuron(X, w, N)

            integer               :: N
            integer, dimension(N) :: X
            real, dimension(N)    :: w
            integer :: neuron, theta

            theta = weighted_sum(X, w, N)
            neuron = activation(theta)

        end function neuron

        ! Weighted sum --------------------------------------------------------

        function weighted_sum(X, w, N)
            
            integer               :: N
            integer, dimension(N) :: X
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

        subroutine adjust_weights(X, w1, w2, w3, N, y1, y2, y, t, a)

            integer               :: N
            integer, dimension(N) :: X, y1, y2
            real, dimension(N,N)  :: w1, w2
            real, dimension(N)    :: w3
            integer :: y, t, i
            real                  :: a
            real, dimension(N)    :: error2
            real, dimension(N,N)  :: error1

            do i = 1, N
                error2(i) = (y - t) * w3(i)
                w3(i) = w3(i) - a * (y - t) * y1(i)
            end do

            do i = 1, N
                error1(i,:) = error2(i) * w2(i,:)
                w2(i,:) = w2(i,:) - a * error2(i) * y1
            end do

            do i = 1, N
                w1(i,:) = w1(i,:) - a * error1(i,:) * X
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