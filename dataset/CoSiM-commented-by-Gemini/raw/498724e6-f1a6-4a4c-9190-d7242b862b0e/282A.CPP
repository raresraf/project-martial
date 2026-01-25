#include <iostream>
#include <string>

using namespace std;

int main() {
  int x = 0;

  int n;
  cin >> n;

  string line;
  for (int i = 0; i < n; i++) {
    cin >> line;
    if (line.find("++") != string::npos) {
      x++;
    } else {
      x--;
    }
  }

  cout << x << endl;
}
