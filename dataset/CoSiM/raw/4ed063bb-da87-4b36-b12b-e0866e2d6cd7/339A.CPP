#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string input;
  getline(cin, input);

  int arr_size = (input.length() + 1) / 2;
  int arr[arr_size];

  int j = 0;
  for (int i = 0; i < input.length(); i = i + 2) {
    arr[j] = int(input.at(i)) - 48;
    j++;
  }

  sort(arr, arr + arr_size);

  for (int j = 0; j < arr_size - 1; j++) {
    cout << arr[j] << '+';
  }

  cout << arr[arr_size - 1];

  return 0;
}
