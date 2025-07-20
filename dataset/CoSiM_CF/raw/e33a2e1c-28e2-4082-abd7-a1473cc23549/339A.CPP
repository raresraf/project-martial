#include <algorithm>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;
  int n = (s.length() / 2 + 1);
  int arr[n];
  for (int i = 0, j = 0; i < s.length(); i += 2, j++)
    arr[j] = (int)s[i] - 48;
  sort(arr, arr + n);
  for (int i = 0; i < n - 1; i++)
    cout << arr[i] << "+";
  cout << arr[n - 1];
  return 0;
}
