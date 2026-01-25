#include <iostream>

using namespace std;
bool can_be_implemented(int *);

int main(void) {
  int n;
  cin >> n;

  // define array
  int data[n][3];

  // take input
  for (int i = 0; i < n; i++)
    cin >> data[i][0] >> data[i][1] >> data[i][2];

  // process data
  int implement_solution_count = 0;

  for (int i = 0; i < n; i++) {
    if (can_be_implemented(data[i]))
      implement_solution_count++;
  }

  cout << implement_solution_count << endl;

  return 0;
}

/* function : can_be_implemented

    * takes an address of an array containing 3 elements
    * returns true if two or more values are 1
    * returns false if two or more values are 0
*/

bool can_be_implemented(int *arr) {
  int one_count = 0;
  for (int i = 0; i < 3; i++)
    if (arr[i] == 1)
      one_count++;

  if (one_count >= 2)
    return true;
  else
    return false;
}