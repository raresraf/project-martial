#include <iostream>
#include <queue>
#include <stdlib.h>
using namespace std;

int main() {
  string s;
  int i = 0;
  priority_queue<int, vector<int>, greater<int>> pq; // To create a min heap
  cin >> s;
  while (s[i] != '\0') {
    if (s[i] != '+')
      pq.push((int)s[i] - 48);
    i++;
  }
  while (!pq.empty()) {
    cout << pq.top();
    if (pq.size() != 1)
      cout << "+";
    pq.pop();
  }
  return 0;
}
