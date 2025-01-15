#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;

  string line;
  cin >> line;

  int cnt = 0;
  for (int i = 1; i < n; i++) {
    if (line[i] == line[i - 1]) {
      cnt++;
    }
  }

  cout << cnt << endl;
}
