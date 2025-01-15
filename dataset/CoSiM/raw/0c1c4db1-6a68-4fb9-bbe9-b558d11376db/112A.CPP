#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  if (n % 7 == 0 || n % 4 == 0 || n % 47 == 0) {
    cout << "YES";
    return 0;
  }
  while (n != 0) {
    int temp = n % 10;
    if (temp != 7 && temp != 4) {
      cout << "NO";
      return 0;
    }
    n = n / 10;
  }
  cout << "YES";
}