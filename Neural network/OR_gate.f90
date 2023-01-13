program OR_gate

    implicit none

    ! Constants ---------------------------------------------------------------

    integer, parameter    :: N = 2 ! Number of inputs
    real                  :: a=0.1 ! Learning rate
    integer, dimension(2**N) :: truth_table = [0, 1, 1, 1]

    ! Variables ---------------------------------------------------------------

    integer, dimension(N) :: X     ! Input
    real, dimension(N)    :: w     ! Weights
    integer               :: y, t  ! Output
    integer               :: i,j   ! Loop counter
    integer               :: tmp   ! Temporary variable
    real                  :: r     ! Random number

    ! Initialize the weights --------------------------------------------------

    do i=1, N
        call random_number(w(i))
    end do

    ! Train the OR gate -------------------------------------------------------

    do i=1, 10000
        ! Draw a random X from {0, 1}^N using uniform distribution
        do j=1, N
            call random_bool(X(j))
        end do

        ! Simulation the neuron
        y = neuron(X, w, N)

        ! convert X to a number
        call bin_to_dec(X, tmp, N)

        ! Compute the target
        t = truth_table(tmp+1)

        ! Adjust the weights
        call adjust_weights(X, w, N, y, t, a)

    end do

    ! Test the OR gate --------------------------------------------------------

    do i=1, N**2
        ! X = i in binary representation
        call dec_to_bin(i-1, X, N)

        ! Simulation the neuron
        y = neuron(X, w, N)

        ! Print the result
        call bin_to_dec(X, tmp, N)
        print *, "Input:", X, "Output:", y, "Expected:", truth_table(tmp+1)

    end do

    ! Print the weights -------------------------------------------------------

    print *, "Weights:", w

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

        subroutine adjust_weights(X, w, N, y, t, a)

            integer               :: N
            integer, dimension(N) :: X
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