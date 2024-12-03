#include<iostream>
#include <array>
#include<algorithm>

int main(void) {
    std::array<int,4> arr = {0,1,2,3};
    std::cout << "Init: " << arr[0] << ", " << arr[1] << ", " << arr[2] << ", " << arr[3] << std::endl;
    int i = 1;
    while (std::next_permutation(arr.begin(), arr.end())) {
        std::cout << "Permutation " << i << ": " << arr[0] << ", " << arr[1] << ", " << arr[2] << ", " << arr[3] << std::endl;
        i++;
    }
    return 0;
}
