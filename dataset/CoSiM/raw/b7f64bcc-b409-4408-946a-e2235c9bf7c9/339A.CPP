#include <algorithm>
#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  string s;
  cin >> s;
  int arr[(s.size())];
  int j = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+') {
      arr[i] = s[i] - 48;
    } else
      arr[i] = 4;
  }
  sort(arr, arr + s.size());
  for (int i = 0; i < s.size(); i++) {
    if (i % 2 == 0)
      cout << arr[i / 2];
    else
      cout << '+';
  }
  return 0;
}