/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Use the same icon import pattern as the rest of the codebase
import {Package} from '@carbon/icons-react';
import OptionButton from './OptionButton';
import useDownloadObjectVideo from './useDownloadObjectVideo';

export default function DownloadObjectOnlyOption() {
  const {downloadObjectOnly, state} = useDownloadObjectVideo();

  // Different loading messages based on the state
  let loadingLabel = 'Processing...';
  if (state === 'downloading') {
    loadingLabel = 'Downloading...';
  }

  return (
    <OptionButton
      title="Download Object Only"
      Icon={Package}
      loadingProps={{
        loading: state === 'started' || state === 'processing' || state === 'downloading',
        label: loadingLabel,
      }}
      onClick={downloadObjectOnly}
    />
  );
}
