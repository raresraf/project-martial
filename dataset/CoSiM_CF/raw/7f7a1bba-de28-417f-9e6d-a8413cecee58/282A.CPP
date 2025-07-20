#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, c = 0;
  cin >> n;
  while (n > 0) {

    char arr[3];
    cin >> arr[0];
    if (arr[0] == '+')
      c++;
    if (arr[0] == '-')
      c--;
    cin >> arr[1];
    cin >> arr[2];
    if (arr[2] == '+')
      c++;
    if (arr[2] == '-')
      c--;

    n--;
  }
  cout << c << endl;

  return 0;
}
