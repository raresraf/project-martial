#include <iostream>

using namespace std;

int main() {
  int n, opinion, sure;
  int problems = 0;

  cin >> n;

  for (int i = 0; i < n; i++) {
    sure = 0;
    for (int j = 0; j < 3; j++) {
      cin >> opinion;
      if (opinion)
        sure++;
    }
    if (sure > 1)
      problems++;
  }

  cout << problems << endl;

  return 0;
}