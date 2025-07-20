// ACCEPTED
#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;
  int total = 0, subtotal = 0;
  int c1 = 0, c2 = 0, c3 = 0;
  for (char a : s) {
    if (a == '1') {
      c1++;
      total++;
    }
    if (a == '2') {
      c2++;
      total++;
    }
    if (a == '3') {
      c3++;
      total++;
    }
  }

  while (c1 != 0) {
    subtotal++;
    if (subtotal != total)
      cout << "1"
           << "+";
    else
      cout << "1";

    c1--;
  }
  while (c2 != 0) {
    subtotal++;
    if (subtotal != total)
      cout << "2"
           << "+";
    else
      cout << "2";

    c2--;
  }
  while (c3 != 0) {
    subtotal++;
    if (subtotal != total)
      cout << "3"
           << "+";
    else
      cout << "3";

    c3--;
  }
}
