#include <iostream>
#include <string>

// Really bad solution i looped 3 times for no reason dont use this

using namespace std;

int main() {
  int n;
  cin >> n;
  string words[n];
  string done[n];

  for (int i = 0; i < n; i++) {
    cin >> words[i];
  }

  for (int i = 0; i < n; i++) {
    if (words[i].length() > 10) {
      done[i] = words[i][0];
      done[i] += to_string(words[i].length() - 2);
      done[i] += words[i][words[i].length() - 1];
    } else {
      done[i] = words[i];
    }
  }

  for (int i = 0; i < n; i++) {
    cout << done[i] << endl;
  }
  return 0;
}
