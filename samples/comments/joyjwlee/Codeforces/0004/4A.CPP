#include <iostream>

using namespace std;
int main() {
  int N;
  cin >> N;

  if (N > 2 && N % 2 == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
