/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 *
 * This program determines if a system of 3D force vectors is in equilibrium.
 * It reads a set of 'n' vectors and checks if their vector sum is the zero
 * vector (0, 0, 0). If it is, the system is in equilibrium, and the program
 * prints "YES"; otherwise, it prints "NO".
 *
 * @algorithm Vector summation.
 * @time_complexity O(N), where N is the number of vectors, as the program
 *                  iterates through the vectors multiple times to sum each
 *                  component.
 * @space_complexity O(N) to store the N vectors in a 2D array.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  /*int **arr=new int*[n];

  for(int i=0;i<n;i++)
  {
          arr=new int[i][3];
  }*/

  // A variable-length array (VLA) is used here, which is a non-standard C++ extension.
  int arr[n][3];

  /**
   * @brief Reads the components of the N vectors.
   * This nested loop populates the 2D array with the x, y, and z components
   * of each of the 'n' vectors.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }

  int sum1 = 0, sum2 = 0, sum3 = 0;
  int flag1 = 0, flag2 = 0, flag3 = 0;

  /**
   * @brief Calculates the sum of the x-components of all vectors.
   */
  for (int i = 0; i < n; i++) {
    sum1 += arr[i][0];
  }

  if (sum1 == 0)
    flag1 = 1;

  /**
   * @brief Conditionally calculates the sum of the y-components.
   * This part of the logic is unnecessarily complex. It calculates the sum of
   * the y-components only if the sum of the x-components is zero.
   */
  if (flag1 == 1) {
    for (int i = 0; i < n; i++) {
      sum2 += arr[i][1];
    }
  }

  else
    sum2 = -9; // Sentinel value to indicate x-component sum was not zero.

  if (sum2 == 0)
    flag2 = 1;

  /**
   * @brief Conditionally calculates the sum of the z-components.
   * Similar to the y-component calculation, this is done only if the sum of
   * the y-components was also zero.
   */
  if (flag2 == 1) {
    for (int i = 0; i < n; i++) {
      sum3 += arr[i][2];
    }
  }

  if (sum3 == 0)
    flag3 = 1;

  /**
   * @brief Checks for equilibrium and prints the result.
   * The system is in equilibrium only if the sums of all three components (x, y, and z) are zero.
   */
  if ((flag1 && flag2) && flag3)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}