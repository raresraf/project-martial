#include <iostream>
#include <string>
using namespace std;

int main() {
  char arr[6] = "hello";
  string chat;
  getline(cin, chat);
  int len = chat.size(), i, j;
  for (i = 0, j = 0; i < len; i++) {
    if (arr[j] == chat[i]) {
      j++;
    }
  }
  if (j == 5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
