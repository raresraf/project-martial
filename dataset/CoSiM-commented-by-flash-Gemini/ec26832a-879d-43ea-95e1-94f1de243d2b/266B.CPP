#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, t;
  string input;

  cin >> n >> t;
  cin >> input;
  vector<char> queue(input.begin(), input.end());

  while (t > 0) {
    for (int i = 1; i < queue.size(); i++) {
      if (queue[i] == 'G' && queue[i - 1] == 'B') {
        swap(queue[i], queue[i - 1]);
        i++;
      }
    }
    t--;
  }

  for (char c : queue) {
    cout << c;
  }
  cout << '\n';
  return 0;
}