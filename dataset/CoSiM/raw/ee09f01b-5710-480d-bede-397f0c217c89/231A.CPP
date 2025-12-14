#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  int count = 0;
  cin >> n;
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;
    if (a + b + c >= 2) {
      count += 1;
    }
  }
  cout << count << endl;
}