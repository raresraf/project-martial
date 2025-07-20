#include <bits/stdc++.h>
using namespace std;
int main() {
  long long n;
  cin >> n;
  int j = 0, arr[100];
  int len = to_string(n).length();
  while (n != 0) {
    arr[len - j - 1] = n % 10;
    n = n / 10;
    j++;
  }
  int i = 0, count = 0;
  while (len) {
    if (arr[i] == 4 || arr[i] == 7)
      count++;
    len--;
    i++;
  }
  if (count == 4 || count == 7)
    cout << "YES";
  else
    cout << "NO";
}
