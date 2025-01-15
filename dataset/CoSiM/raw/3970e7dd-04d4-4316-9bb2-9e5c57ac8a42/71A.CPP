#include <bits/stdc++.h>
using namespace std;
int main() {
  string arr;
  int n;
  cin >> n;

  while (n > 0) {
    cin >> arr;
    if (arr.size() > 10) {
      arr = arr[0] + to_string(arr.size() - 2) + arr[arr.size() - 1];
      cout << arr << endl;
    } else
      cout << arr << endl;

    n--;
  }
}
