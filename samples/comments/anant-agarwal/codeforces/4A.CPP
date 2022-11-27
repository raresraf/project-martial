#include <cstdio>
#include <iostream>
using namespace std;
int main() {
  int a;
  cin >> a;
  if (a % 2 || a / 2 < 2)
    cout << "NO"
         << "\n";
  else
    cout << "YES"
         << "\n";
}