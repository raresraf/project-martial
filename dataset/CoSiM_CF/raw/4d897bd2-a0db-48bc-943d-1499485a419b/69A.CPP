#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(void) {
  int numberOfVectors;
  cin >> numberOfVectors;

  int arr[numberOfVectors][3];

  for (int i = 0; i < numberOfVectors; i++)
    cin >> arr[i][0] >> arr[i][1] >> arr[i][2];

  int p = 0, q = 0, r = 0;
  for (int i = 0; i < numberOfVectors; i++) {
    p += arr[i][0];
    q += arr[i][1];
    r += arr[i][2];
  }

  if (p == 0 && q == 0 && r == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}