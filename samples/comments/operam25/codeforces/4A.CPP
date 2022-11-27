#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  if (n < 4 || n & 1)
    cout << "NO";
  else
    cout << "YES";
}