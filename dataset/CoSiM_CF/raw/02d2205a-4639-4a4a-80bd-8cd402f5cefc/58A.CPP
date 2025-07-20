#include <bits/stdc++.h>
using namespace std;
int main() {

  char A[10] = "hello", a;
  int i = 0;
  while (cin >> a && a != '\n') {
    if (a == A[i])
      i++;
  }
  cout << (i == 5 ? "YES" : "NO");
}