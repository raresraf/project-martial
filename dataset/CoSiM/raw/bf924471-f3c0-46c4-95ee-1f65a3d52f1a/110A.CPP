#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int a, b, c = 0, digit = 0, s = 0, len = 0;
  cin >> a;
  while (a > 0) {
    b = a % 10;
    // cout<<b<<endl;
    if (b == 4 || b == 7) {
      c++;
    }
    a = a / 10;
  }
  // cout<<c<<endl;
  // cout<<digit<<endl;
  b = 0;
  // s=c;
  while (c > 0) {
    b = c % 10;
    // cout<<b<<endl;
    if (b == 4 || b == 7) {
      digit++;
    }
    c = c / 10;
    len++;
  }
  // cout<<s<<endl;
  // cout<<digit<<endl;
  if (len > 0 && digit > 0 && len == digit) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
