#include <iostream>
using namespace std;
int main() {
  long long n;
  cin >> n;
  if ((n % 2) == 0 && n != 2)
    cout << "YES\n";
  else
    cout << "NO\n";
}