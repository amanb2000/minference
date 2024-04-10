import random

class LexFridmanDataLoader:
    def __init__(self, dataset, num_chars_per_chunk, random_seed=None):
        self.dataset = dataset
        self.num_chars_per_chunk = num_chars_per_chunk
        self.current_episode_index = 0
        self.current_char_index = 0
        
        # Seed the random number generator
        random.seed(random_seed)

        
        # Shuffle the dataset
        # Shuffle the dataset
        self.episodes = [episode for episode in self.dataset]
        random.shuffle(self.episodes)
    
    def get_next_chunk(self):
        """
        Returns the next chunk from the current episode. If the end of the episode is reached,
        it moves on to the next episode.
        """
        if self.current_episode_index >= len(self.episodes):
            # All episodes have been processed
            return None
        
        current_episode = self.episodes[self.current_episode_index]
        remaining_chars_in_episode = len(current_episode['text']) - self.current_char_index
        
        if remaining_chars_in_episode == 0:
            # Move on to the next episode
            self.current_episode_index += 1
            self.current_char_index = 0
            return self.get_next_chunk()
        
        # Determine the end index for this chunk
        end_char_index = min(self.current_char_index + self.num_chars_per_chunk, 
                             len(current_episode['text']))
        
        # Extract the chunk
        chunk = current_episode['text'][self.current_char_index:end_char_index]
        self.current_char_index = end_char_index
        
        # Return the chunk along with supplemental information
        supplemental_info = {
            'episode_id': current_episode['id'],
            'episode_title': current_episode['title'],
            'start_char_index': self.current_char_index,
            'end_char_index': end_char_index
        }
        
        return chunk, supplemental_info