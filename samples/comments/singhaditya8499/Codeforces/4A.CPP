#include <iostream>
using namespace std;
int main() {
  // your code goes here
  int a;
  cin >> a;
  if (a % 2 == 0 && a != 2)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}