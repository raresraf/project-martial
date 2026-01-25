#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int count = 0;
  while (n--) {
    int cnt = 0, temp;

    for (int i = 0; i < 3; i++) {
      cin >> temp;
      if (temp == 1)
        cnt++;
    }

    if (cnt >= 2)
      count++;
  }

  cout << count << endl;

  return 0;
}
