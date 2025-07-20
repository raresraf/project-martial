#include <iostream>

using namespace std;

int main() {

  int n;
  cin >> n;
  string line;
  for (int i = 0; i < n; ++i) {
    cin >> line;
    int length = line.length();
    if (length < 11)
      cout << line << endl;
    else
      cout << line[0] << length - 2 << line[length - 1] << endl;
  }

  return 0;
}
