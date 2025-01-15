#include <cstring>
#include <iostream>

using namespace std;

int main() {
  int stone_count;
  cin >> stone_count;

  string stones;
  cin >> stones;

  int count = 0;

  char prev, current;
  for (int i = 0; i + 1 < stone_count; i++) {
    prev = stones[i];
    current = stones[i + 1];
    if (prev == current) {
      count++;
    }
  }

  cout << count;
  return 0;
}